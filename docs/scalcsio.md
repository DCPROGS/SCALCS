# scalcsio — File I/O

`scalcs.scalcsio`

Reading DCPROGS binary files: mechanism files (`.mec`), single-channel
records (`.scn`), and amplitude/time files (`.ssd`, `.abf`).

---

## MEC files (kinetic mechanism)

DCPROGS `.mec` files store one or more kinetic mechanisms (rate sets).

### `mec_get_list(mecfile)`

Read the list of all mechanisms stored in a `.mec` file.

```python
version, meclist, max_mecnum = mec_get_list('channel.mec')
```

| Return | Type | Description |
|--------|------|-------------|
| `version` | int | File format version (latest: 102) |
| `meclist` | list | One entry per rate set; each entry is `[jstart, mecnum, mectitle, ratetitle]` |
| `max_mecnum` | int | Number of distinct mechanisms |

### `mec_load(mecfile, mec_idx)`

Load a specific mechanism (by index into `meclist`) and return a
`Mechanism` object.

```python
from scalcs import scalcsio
version, meclist, n = scalcsio.mec_get_list('channel.mec')
mec = scalcsio.mec_load('channel.mec', meclist[0])
```

---

## SCN files (single-channel record)

DCPROGS `.scn` files contain idealised single-channel dwell times and
amplitude levels produced by the SCAN program.

### `scn_read(scnfile)`

Read a `.scn` file and return arrays of dwell times and amplitude levels.

```python
times, amps, iclass = scn_read('record.scn')
```

| Return | Description |
|--------|-------------|
| `times` | Dwell times (s) |
| `amps` | Amplitude levels (pA or normalised) |
| `iclass` | Event classification codes |

---

## Usage example

```python
from scalcs import scalcsio

# List available mechanisms
version, meclist, n = scalcsio.mec_get_list('my_channel.mec')
for entry in meclist:
    print(entry[2], entry[3])   # title, rate set name

# Load the first mechanism
mec = scalcsio.mec_load('my_channel.mec', meclist[0])
mec.set_eff('c', 1e-6)
```

---

## Notes

- Files are read in binary mode using Python's `struct` and `array` modules
  to match the DCPROGS Fortran binary layout.
- File version 102 is the current DCPROGS standard; older versions (100, 101)
  are also supported.
