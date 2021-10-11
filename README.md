# SCOTER

[![DOI](https://img.shields.io/badge/DOI-10.5880%2FGFZ.2.1.2019.002-blue.svg)](http://doi.org/10.5880/GFZ.2.1.2019.002)
[![](https://img.shields.io/badge/licence-GPL--3.0-orange)](LICENSE)

**SCOTER** implements static and shrinking-box source-specific station terms
(SSST) techniques to reduce the effect of spatially correlated residuals
caused by 3-D velocity structure and improve the relative location accuracy
among nearby seismic events.

## Citation

The recommended citation for SCOTER is:

>  Nooshiri, Nima; Heimann, Sebastian; Tilmann, Frederik; Dahm, Torsten; Saul, 
Joachim (2019): SCOTER - Software package for multiple-earthquake relocation by 
using static and source-specific station correction terms. V. 0.1. GFZ Data Services. http://doi.org/10.5880/GFZ.2.1.2019.002

## License

GNU General Public License, Version 3, 29 June 2007

Copyright © 2019 Helmholtz Centre Potsdam - GFZ German Research Centre for
Geosciences, Potsdam, Germany

SCOTER is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. SCOTER is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

## Contact
* Nima Nooshiri - nima.nooshiri@gfz-potsdam.de

```
GFZ German Research Centre for Geosciences
Section 2.1: Physics of Earthquakes and Volcanoes
Helmholtzstr. 6/7
14467 Potsdam, Germany
```

## Download and Installation

1. Install [Pyrocko](https://pyrocko.org/):
    See Pyrocko installation page [here](https://pyrocko.org/docs/current/install/).

2. Install NonLinLoc:
    See NonLinLoc repository that is modified and packaged as SCOTER backend [here](https://gitext.gfz-potsdam.de/nooshiri/scoter-nonlinloc.git).

3. Install SCOTER and the rest of dependencies:

    ```bash
    cd ~/src/   # or wherever you keep your source packages
    git clone https://gitext.gfz-potsdam.de/nooshiri/scoter.git
    cd scoter
    sudo python setup.py install
    ```

## Documentation

The SCOTER download includes a users guide, including examples that describes
rudimentary usage. Input and output file formats are specified.
