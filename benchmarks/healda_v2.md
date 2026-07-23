# HealDA-v2 inference

HealDA-v2 currently uses random weights and generated normalization, PCA, and
static-conditioning assets. These examples exercise the inference pipeline but
do not produce a scientifically valid analysis.

## Installation

Install the HealDA-v2 extra from the repository:

```bash
pip install -e ".[da-healda-v2]"
```

The extra installs the pinned PhysicsNeMo Git revision required for
`VideoHealDA` and context-parallel inference. PhysicsNeMo 2.1.1 does not
provide HealDA-v2.

## Real UFS replay observations

Save this as `run_healda_v2_real.py`:

```python
import numpy as np

from earth2studio.data import UFSObsConv, UFSObsSat, fetch_dataframe
from earth2studio.models.da.healda_v2 import HealDAv2

model_parallel_size = 1
analysis_time = np.datetime64("2024-01-01T00:00")
request_time = np.array([analysis_time], dtype="datetime64[ns]")

model = HealDAv2.load_model(
    model_parallel_size=model_parallel_size,
    lat_lon=True,
)
time_tolerance = model.input_time_tolerance(analysis_time)
conv_schema, sat_schema = model.input_coords()

conv_df = fetch_dataframe(
    UFSObsConv(time_tolerance=time_tolerance),
    time=request_time,
    variable=np.asarray(conv_schema["variable"]),
    fields=np.asarray(list(conv_schema.keys())),
)
sat_df = fetch_dataframe(
    UFSObsSat(time_tolerance=time_tolerance),
    time=request_time,
    variable=np.asarray(sat_schema["variable"]),
    fields=np.asarray(list(sat_schema.keys())),
)

result = model(
    conv_obs=conv_df,
    sat_obs=sat_df,
    analysis_time=analysis_time,
)
print(result)
```

Set `model_parallel_size` above to match `--nproc-per-node`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 \
  EARTH2STUDIO_CACHE="$PWD/.cache/earth2studio" \
  torchrun --standalone --nproc-per-node=1 run_healda_v2_real.py

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  EARTH2STUDIO_CACHE="$PWD/.cache/earth2studio" \
  torchrun --standalone --nproc-per-node=4 run_healda_v2_real.py
```

## Generated observations

```python
import numpy as np

from benchmarks.healda_v2_end_to_end import make_random_observations
from earth2studio.models.da.healda_v2 import HealDAv2

model_parallel_size = 1
analysis_time = np.datetime64("2024-01-01T00:00")
model = HealDAv2.load_model(
    model_parallel_size=model_parallel_size,
    lat_lon=True,
)
conv_df, sat_df = make_random_observations(
    model,
    analysis_time,
    conv_obs_per_frame=1_250_000,
    sat_obs_per_frame=1_250_000,
)

result = model(
    conv_obs=conv_df,
    sat_obs=sat_df,
    analysis_time=analysis_time,
)
print(result)
```

Set `model_parallel_size` above to match `--nproc-per-node`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 \
  torchrun --standalone --nproc-per-node=1 run_healda_v2_random.py

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --standalone --nproc-per-node=4 run_healda_v2_random.py
```

The timed benchmark provides the same two data modes:

```bash
# Real UFS replay observations
torchrun --standalone --nproc-per-node=4 \
  benchmarks/healda_v2_end_to_end.py --model-parallel-size=4

# 1.25M conventional and 1.25M satellite observations per frame
torchrun --standalone --nproc-per-node=4 \
  benchmarks/healda_v2_end_to_end.py --model-parallel-size=4 --random-data
```

The trained HealDA-v1 example with UFS fetching and ERA5 MAE is
`examples/05_data_assimilation/02_healda.py`.
