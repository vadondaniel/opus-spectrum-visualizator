# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

# adding this file to here makes processing start faster
"""Extend NDDataset with the import method for OPUS generated data files."""

__all__ = ["read_opus"]
__dataset_methods__ = __all__

from datetime import datetime
from datetime import timedelta
import re

import numpy as np

from spectrochempy.application.application import debug_
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.readers.importer import _openfid
from spectrochempy.extern.brukeropus import OPUSFile
from spectrochempy.extern.brukeropus.file.utils import get_block_type_label
from spectrochempy.extern.brukeropus.file.utils import get_param_label
from spectrochempy.utils.datetimeutils import UTC
from spectrochempy.utils.docutils import docprocess

# from spectrochempy.application.application import warning_
from spectrochempy.utils.meta import Meta

# ======================================================================================
# Public functions
# ======================================================================================
docprocess.delete_params("Importer.see_also", "read_opus")


@docprocess.dedent
def read_opus(*paths, **kwargs):
    r"""
    Open Bruker OPUS file(s).

    Eventually group them in a single dataset. Returns an error if dimensions are incompatibles.


    Parameters
    ----------
    %(Importer.parameters)s
    type : str, optional
        The type of data to be read. Possible values are:

        - "AB": Absorbance (default if present in the file)
        - "TR": Transmittance
        - "KM": Kubelka-Munk
        - "RAM": Raman
        - "EMI": Emission
        - "RFL": Reflectance
        - "LRF": log(Reflectance)
        - "ATR": ATR
        - "PAS": Photoacoustic
        - "RF": Single-channel reference spectra
        - "SM": Single-channel sample spectra
        - "IGRF": Reference interferogram
        - "IGSM": Sample interferogram
        - "PHRF": Reference phase
        - "PHSM": Sample phase

        An error is raised if the specified type is not present in the file.

    Returns
    -------
    %(Importer.returns)s

    Other Parameters
    ----------------
    %(Importer.other_parameters)s

    See Also
    --------
    %(Importer.see_also.no_read_opus)s

    Examples
    --------
    Reading a single OPUS file  (providing a windows type filename relative to
    the default `datadir` )

    >>> scp.read_opus('irdata\\OPUS\\test.0000')
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Reading a single OPUS file  (providing a unix/python type filename relative to the
    default `datadir` )

    >>> scp.read_opus('irdata/OPUS/test.0000')
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Single file specified with pathlib.Path object

    >>> from pathlib import Path
    >>> folder = Path('irdata/OPUS')
    >>> p = folder / 'test.0000'
    >>> scp.read_opus(p)
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Multiple files not merged (return a list of datasets). Note that a directory is
    specified

    >>> le = scp.read_opus('test.0000', 'test.0001', 'test.0002',
    >>>                    directory='irdata/OPUS')
    >>> len(le)
    3
    >>> le[0]
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Multiple files merged as the `merge` keyword is set to true

    >>> scp.read_opus('test.0000', 'test.0001', 'test.0002',
    directory='irdata/OPUS', merge=True)
    NDDataset: [float64] a.u. (shape: (y:3, x:2567))

    Multiple files to merge : they are passed as a list instead of using the keyword `
    merge`

    >>> scp.read_opus(['test.0000', 'test.0001', 'test.0002'],
    >>>               directory='irdata/OPUS')
    NDDataset: [float64] a.u. (shape: (y:3, x:2567))

    Multiple files not merged : they are passed as a list but `merge` is set to false

    >>> le = scp.read_opus(['test.0000', 'test.0001', 'test.0002'],
    >>>                    directory='irdata/OPUS', merge=False)
    >>> len(le)
    3

    Read without a filename. This has the effect of opening a dialog for file(s)
    selection

    >>> nd = scp.read_opus()

    Read in a directory (assume that only OPUS files are present in the directory
    (else we must use the generic `read` function instead)

    >>> le = scp.read_opus(directory='irdata/OPUS')
    >>> len(le)
    4

    Again we can use merge to stack all 4 spectra if thet have compatible dimensions.

    >>> scp.read_opus(directory='irdata/OPUS', merge=True)
    NDDataset: [float64] a.u. (shape: (y:4, x:2567))

    Bruker OPUS files often contain several types of data (AB, RF, IGSM ...).
    In some cases, the type of data can be inferred from the file content.
    For instance, if the file contains a single background spectrum, it is inferred
    as a reference spectrum. In the following example, the file `background.0` is
    correctly inferred as a reference spectrum.

    >>> B = scp.read_opus('irdata/OPUS/background.0')
    >>> B
    NDDataset: [float64] a.u. (shape: (y:1, x:4096))

    If the type of data can not be inferred, an error is raised. In the following
    example, if the file `test.0000` contains only sample spectra (SM), the type of data
    must be specified.

    >>> A = scp.read_opus('irdata/OPUS/test.0000', type='SM')
    >>> A
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    """
    kwargs["filetypes"] = ["Bruker OPUS files (*.[0-9]*)"]
    kwargs["protocol"] = ["opus"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================
# Private Functions
# ======================================================================================

_data_types = {
    "sm": "Single-channel sample spectra",
    "rf": "Single-channel reference spectra",
    "igsm": "Sample interferogram",
    "igrf": "Reference interferogram",
    "phsm": "Sample phase",
    "phrf": "Reference phase",
    "a": "Absorbance",
    "t": "Transmittance",
    "r": "Reflectance",
    "km": "Kubelka-Munk",
    "tr": "Trace (Intensity over Time)",
    "gcig": "gc File (Series of Interferograms)",
    "gcsc": "gc File (Series of Spectra)",
    "ra": "Raman",
    "e": "Emission",
    "pw": "Power",
    "logr": "log(Reflectance)",
    "atr": "ATR",
    "pas": "Photoacoustic",
}

_types_parameters = {
    "RF": "rf",
    "SM": "sm",
    "IGRF": "igrf",
    "IGSM": "igsm",
    "PHRF": "phrf",
    "PHSM": "phsm",
    "AB": "a",
    "TR": "t",
    "KM": "km",
    "RAM": "ra",
    "EMI": "e",
    "RFL": "r",
    "LRF": "logr",
    "ATR": "atr",
    "PAS": "pas",
}

_types_parameters_inv = {v: k for k, v in _types_parameters.items()}

_blocks_parameters = {
    0: "",
    1: "Data Status",  # 'Data Status Parameters',
    2: "Instrument Status",  # 'Instrument Status Parameters',
    3: "Acquisition",  # 'Acquisition Parameters',
    4: "Fourier Transform",  # 'Fourier Transform Parameters',
    5: "Plot and Display",  # 'Plot and Display Parameters',
    6: "Optical",  # 'Optical Parameters',
    7: "GC",  # 'GC Parameters',
    8: "Library Search",  # 'Library Search Parameters',
    9: "Communication",  # 'Communication Parameters',
    10: "Sample Origin",  # 'Sample Origin Parameters',
    11: "Lab and Process",  # 'Lab and Process Parameters',
}

_blocks_parameters_inv = {v: k for k, v in _blocks_parameters.items()}

_units = {
    "WN": ("wavenumber", "cm^-1"),
    "WL": ("wavelength", "µm"),
    "Absorbance": "absorbance",  # defined in core/units
    "Transmittance": "transmittance",
    "Kubelka-Munk": "Kubelka_Munk",
}

_HISTORY_TIMESTAMP_RE = re.compile(
    r"(?P<date>(?:\d{4}/\d{2}/\d{2}|\d{2}/\d{2}/\d{4}))\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*"
    r"\(GMT\s*(?P<offset>[+-](?:\d+(?:\.\d+)?|\d{1,2}:\d{2}))\)"
)


def _get_timestamp_from(params):
    # get the acquisition timestamp
    acqdate = params.dat
    acqtime = params.tim
    gmt_offset_hour = float(acqtime.split("GMT")[1].split(")")[0])
    if len(acqdate.split("/")[0]) == 2:
        formats = ("%d/%m/%Y_%H:%M:%S.%f", "%d/%m/%Y_%H:%M:%S")
    elif len(acqdate.split("/")[0]) == 4:
        formats = ("%Y/%m/%d_%H:%M:%S.%f", "%Y/%m/%d_%H:%M:%S")
    else:  # pragma: no cover
        raise ValueError("acqdate can not be interpreted.")

    date_time = None
    raw_dt = acqdate + "_" + acqtime.split()[0]
    for fmt in formats:
        try:
            date_time = datetime.strptime(raw_dt, fmt)
            break
        except ValueError:
            continue
    if date_time is None:
        raise ValueError("acqtime can not be interpreted.")

    utc_dt = date_time - timedelta(hours=gmt_offset_hour)
    utc_dt = utc_dt.replace(tzinfo=UTC)
    return utc_dt, utc_dt.timestamp()


def _parse_gmt_offset_hours(offset_text):
    if ":" in offset_text:
        sign = -1 if offset_text.startswith("-") else 1
        hour_text, minute_text = offset_text[1:].split(":", 1)
        return sign * (int(hour_text) + int(minute_text) / 60.0)
    return float(offset_text)


def _parse_opus_datetime(date_text, time_text):
    dt_text = f"{date_text} {time_text}"
    formats = (
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
    )
    for fmt in formats:
        try:
            return datetime.strptime(dt_text, fmt)
        except ValueError:
            continue
    return None


def _get_first_history_timestamp(opus_history):
    if not opus_history:
        return None, None

    timestamps_utc = []
    for match in _HISTORY_TIMESTAMP_RE.finditer(opus_history):
        local_dt = _parse_opus_datetime(match.group("date"), match.group("time"))
        if local_dt is None:
            continue
        offset_hours = _parse_gmt_offset_hours(match.group("offset"))
        utc_dt = (local_dt - timedelta(hours=offset_hours)).replace(tzinfo=UTC)
        timestamps_utc.append(utc_dt)

    if not timestamps_utc:
        return None, None

    first_dt = min(timestamps_utc)
    return first_dt, first_dt.timestamp()


def _get_first_modified_timestamp(opus_data, fallback_params):
    first_dt, first_timestamp = _get_first_history_timestamp(
        getattr(opus_data, "history", None)
    )
    if first_timestamp is not None:
        return first_dt, first_timestamp, "history"

    try:
        fallback_dt, fallback_timestamp = _get_timestamp_from(fallback_params)
        return fallback_dt, fallback_timestamp, "acquisition"
    except Exception:
        return None, None, "missing"


def _load_parameters_into_meta(opus_data):
    # Load the parameters in the metadata.
    meta = Meta()
    param_infos = [
        ("Sample/Result Parameters", "params"),
        ("Reference Parameters", "rf_params"),
    ]
    for title, attr in param_infos:
        meta[attr] = Meta(name=title)
        blocks = getattr(opus_data, attr).blocks
        for block in blocks:
            id = f"{_blocks_parameters[block.type[2]].lower().replace(' ', '_')}"
            meta[attr][id] = Meta(name=get_block_type_label(block.type))
            for key in block.keys:
                name = get_param_label(key)
                value = getattr(getattr(opus_data, attr), key)
                meta[attr][id][key] = Meta(name=name, value=value)

    return meta


@_importer_method
def _read_opus(*args, **kwargs):
    debug_("Bruker OPUS import")

    dataset, filename = args

    fid, kwargs = _openfid(filename, **kwargs)

    opus_data = OPUSFile(fid)

    # Which data are present in the file?
    all_data_types = opus_data.all_data_keys  # type of data present in the file
    possible_type_parameters = [_types_parameters_inv[k] for k in all_data_types]

    # Get type parameter
    type_parameter = kwargs.get("type")

    # Check if the data type is specified
    if type_parameter is None:
        # if AB in possible_type_parameters, take it as default (backward compatibility)
        if "AB" in possible_type_parameters:
            type_parameter = "AB"

        # if RAM, it is a raman spectra
        elif "RAM" in possible_type_parameters:
            type_parameter = "RAM"

        # if EMI, it is an emission spectra
        elif "EMI" in possible_type_parameters:
            type_parameter = "EMI"

        # it may be a single background
        elif possible_type_parameters == ["RF", "IGRF"]:
            type_parameter = "RF"

        else:
            raise ValueError(
                f"Please specify the type of data to read. Possible values for this file are: {possible_type_parameters}"
            )
        # warning_(
        #     f"Default type parameter {type_parameter} is used.\n"
        #     "Please specify the type of data to read if this is not the correct data type to read.\n"
        #     f"Possible values for this file are: {possible_type_parameters}"
        # )

    elif type_parameter is not None and type_parameter not in possible_type_parameters:
        raise ValueError(
            f"The data type {type_parameter} is not present in the file. Possible values for this file are: {possible_type_parameters}"
        )

    data_type = _types_parameters[type_parameter]

    d = getattr(opus_data, data_type)

    # data
    dataset.data = d.y if d.y.ndim > 1 else d.y[np.newaxis]
    desc = _data_types[data_type]
    dataset.title = desc.lower()
    dataset.units = _units.get(desc)  # None if not found in _units

    # xaxis
    title, units = _units.get(getattr(d.params, "dxu"))  # noqa: B009
    xaxis = Coord(d.x, title=title, units=units)

    # yaxis (in case this is not a data series)
    # TODO: check if this is a data series and read eventually 2D data
    name = opus_data.params.snm
    dt, timestamp = _get_timestamp_from(d.params)

    yaxis = Coord(
        [timestamp],
        title="acquisition timestamp (GMT)",
        units="s",
        labels=([dt], [name], [filename]),
    )

    # set dataset's Coordset
    dataset.set_coordset(y=yaxis, x=xaxis)

    # Set name, origin, description and history
    dataset.name = filename.stem
    dataset.filename = filename
    dataset.origin = f"opus-{type_parameter}"
    dataset.description = "Dataset from opus files. \nSpectra type: " + desc
    dataset.history = str(datetime.now(UTC)) + ": import from opus files \n"

    # add other parameters in metadata
    dataset.meta = _load_parameters_into_meta(opus_data)
    dataset.meta.name = "OPUS Parameters"
    first_dt, first_timestamp, first_source = _get_first_modified_timestamp(
        opus_data, d.params
    )
    dataset.meta["first_modified_timestamp"] = first_timestamp
    dataset.meta["first_modified_datetime_utc"] = (
        first_dt.isoformat() if first_dt is not None else None
    )
    dataset.meta["first_modified_source"] = first_source

    # add info about other type present in th file and which could be alternatively read.
    dataset.meta["other_data_types"] = possible_type_parameters

    # set the dataset as readonly now that all the metadata is set
    dataset.meta.readonly = True

    # reset modification date to cretion date
    dataset._modified = dataset._created

    return dataset
