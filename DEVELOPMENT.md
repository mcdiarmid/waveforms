# Developer notes

## Setting-up an Environment

### With pip and venv

```bash
python -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
pip install -e waveforms-*  # Currently required for editable sub-modules, need to fx this
```
