# -*- coding: utf-8 -*-

import glob
import os
import os.path as op
from string import Template
from subprocess import PIPE, Popen
from tempfile import NamedTemporaryFile

from pyrocko.util import ensuredir
import progressbar as pb_mod

from .log_util import custom_logger
from .meta import expand_template
from .parmap import parstarmap


SUFFIX = '-nllocrun-{}'.format(os.environ['USER'])

# Set logger.
pb_mod.streams.wrap_stderr()
logger = custom_logger(__name__)

# How to call the programs.
program_bins = {'nlloc': 'NLLoc'}

# LOCFILES template string.
fline_template = Template(
    'LOCFILES ${obsfile} NLLOC_OBS ${ttpath} ${outroot} ${swapbytes_flag}')

# Directory of temporary files.
if (os.environ.get('SSH_CONNECTION', None) or
        os.environ.get('SSH_CLIENT', None)):
    # Remote login by user. Access to `/tmp` can be slow.
    tmp_dir = op.join(op.curdir, '.tmp'+SUFFIX)
    ensuredir(tmp_dir)
else:
    tmp_dir = None


class NLLocError(Exception):
    pass


g_state = {}


def nlloc_runner(config, keep_scat=False, raise_exception=False):
    """
    Main work routine of the NLLoc program.

    Parameters
    ----------
    config : :class:`scoter.core.Config` object
        SCOTER main configuration.

    keep_scat : bool (default: False)
        Whether to keep sactter files (*.scat and *.hdr) or delete them.

    raise_exception : bool (default: False)
        If True then raise an exception if NLLoc has a non-zero exit state.
    """

    g_state[id(config)] = (config, keep_scat, raise_exception)

    ntargets = len(config.targets)
    task_list = zip(xrange(ntargets), [id(config)]*ntargets)

    # Run the processes.
    ensuredir(config.locdir)

    logger.info('Locating seismic events')

    _ = parstarmap(   # noqa
        _nlloc_worker,
        task_list,
        nparallel=config.nparallel,
        show_progress=config.show_progress,
        label=None)


def _nlloc_worker(itarget, config_id):
    """
    Locate a `single` event by calling `NLLoc` program and save
    the output files with the same name as the input file.

    This function is used in *parallel* processing.

    Parameters
    ----------
    icontrol_file : str
        Name of :class:`tempfile.NamedTemporaryFile` file which is
        infact the control file of an individual event. This file is
        removed after finishing the process.

    outroot : str
        Full path and file root name of the output files. Used to find
        and remove redundant and temporary files, as well as renaming the
        NLLoc output location file.

    Note
    ----
    The given *control file* is deleted after finishing the process.
    """

    config, keep_scat, raise_exception = g_state[config_id]

    target = config.targets[itarget]

    obsfile = expand_template(
        config.dataset_config.bulletins_template_path,
        dict(event_name=target.name))
    outroot = op.join(config.locdir, target.name)
    ttpath = config.dataset_config.traveltimes_path

    fline = fline_template.substitute(
        obsfile=obsfile,
        ttpath=ttpath,
        outroot=outroot,
        swapbytes_flag=config.nlloc_config._swapbytes_flag)

    with NamedTemporaryFile(
            mode='w+t', suffix=SUFFIX, delete=True, dir=tmp_dir) as f:

        f.write('{}\n'.format(config.nlloc_config))
        f.write('{}\n'.format(fline))

        # Write station delays.
        if isinstance(target.station_delays, basestring):
            # Given station delays as a plain text file.
            f.write('INCLUDE {}\n'.format(target.station_delays))
        elif isinstance(target.station_delays, list):
            for delay in target.station_delays:
                f.write('{}\n'.format(delay))

        # Write stations.
        if target.station_labels:
            for slabel in target.station_labels:
                try:
                    sta = [s for s in config.stations if s.name == slabel][0]
                except IndexError:
                    continue
                else:
                    f.write('{}\n'.format(sta))
        else:
            f.write(config.stations_stream)

        # Return the pointer to the top of the file before reading it.
        # Otherwise, just an empty string will be read.
        f.seek(0)

        # Run NLLoc.
        program = program_bins['nlloc']

        try:
            proc = Popen([program, f.name], stdout=PIPE, stderr=PIPE)
        except OSError:
            raise NLLocError('could not start program: "{}"'.format(program))

        nlloc_out, nlloc_err = proc.communicate()

    logger.debug(
        "======= BEGIN NLLoc OUTPUT =======\n{}"
        "======= END NLLoc OUTPUT =======".format(nlloc_out))
    if nlloc_err:
        logger.error(
            "======= BEGIN NLLoc ERROR =======\n{}"
            "======= END NLLoc ERROR =======".format(nlloc_err))
    errmess = []
    if proc.returncode != 0:
        errmess.append(
            'NLLoc had a non-zero exit state: {}'.format(proc.returncode))
    if nlloc_err:
        errmess.append('NLLoc emitted something via stderr')
    if nlloc_err.lower().find('error') != -1:
        errmess.append("The string 'error' appeared in NLLoc output")
    if errmess and raise_exception:
        raise NLLocError(
            "===== BEGIN NLLoc OUTPUT =====\n{0}\n"
            "===== END NLLoc OUTPUT =====\n"
            "===== BEGIN NLLoc ERROR =====\n{1}\n"
            "===== END NLLoc ERROR =====\n"
            "{2}\n"
            "NLLoc HAS BEEN INVOKED AS '{3}'".format(
                nlloc_out,
                nlloc_err,
                '\n'.join(errmess),
                program))

    if nlloc_out.lower().find('0 events located') != -1:
        logger.warn('NLLoc skipped locating event: "{}"'.format(target.name))
    else:
        # Location completed.
        # Step 1: rename the output files.
        # path-Name.date.time.gridN.loc.FileExtension (see NonLinLoc docs).
        if keep_scat:
            for ext in ['hyp', 'hdr', 'scat']:
                oldfile = glob.glob(outroot + '.[0-9]*.[0-9]*.loc.' + ext)[0]
                newfile = outroot + '.loc.' + ext
                os.rename(oldfile, newfile)
        else:
            oldfile = glob.glob(outroot + '.[0-9]*.[0-9]*.loc.' + 'hyp')[0]
            newfile = outroot + '.loc.' + 'hyp'
            os.rename(oldfile, newfile)

            for ext in ['hdr', 'scat']:
                f = glob.glob(outroot + '.[0-9]*.[0-9]*.loc.' + ext)[0]
                os.remove(f)

    # Step 2: delete redundant files (summary & temporary files).
    # path-Name.sum.gridN.loc.FileExtension (see NonLinLoc docs)
    for sf in glob.glob(outroot + '.sum.grid*.loc.*'):
        os.remove(sf)
    for tf in glob.glob(outroot + '*' + SUFFIX):
        os.remove(tf)


__all__ = ['nlloc_runner']
