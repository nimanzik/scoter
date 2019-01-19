To use SCOTER with your own dataset, we suggest the following folder structure.
For detailed instructions see the SCOTER documentation.

project-dir/
    ├─ config/               % contains configuration file
    │    └─ config_file.sf
    ├─ data/
    │    ├─ phase/           % contains observation phase files
    │    │    ├─ eid_001.nll
    │    │    ├─ eid_002.nll
    │    │    └─ ...
    │    └─ events.pf        % catalogue information about the events
    ├─ meta/                 % contains one or more station files
    │    ├─ stations.sf
    │    └─ ...
    ├─ time/                 % contains travel-time grid files
    │    ├─ ak135.Pn.DEFAULT.time.buf
    │    ├─ ak135.Pn.DEFAULT.time.hdr
    │    └─ ...
    └─ run_dir/              % created at runtime, contains relocation results

