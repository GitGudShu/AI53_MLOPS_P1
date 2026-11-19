# AI53 | MLOps : Project Phase 1
## From an Image Experiment to a Reproducible Training Pipeline

### Milestone 0 - Audit the start before changing it

6 issues to explain and fix as per milestone 0. Each row below describes a reproducibility problem, its location in the starter code, and a suggested fix.

| File / Setting | Problem | Suggested Fix |
| --- | --- | --- |
| `data/raw` | The images aren't versionned meaning git doesn't track the data, which is normal but if the archive changes upstream no one will know. | We need to track the data with `dvc` with a DVC remote. |
| `src/reproducibility.py` used in `data.py` | there is a random seed but it's not seeded for numpy, torch, cuda etc... (not implemented) | Well we'd have to implement it and to test it works, there's `test/reproducibility.py` for that. |
| `src/prepare.py` line `33` | Again no `dvc.yaml`, nobody can tell which outputs are stale after `params.yaml` or the data changes. | We need to implement a `dvc.yaml` file with `deps`, `params`, and `outs`. We can verify with `dvc status` after editing a parameter. |
| `src/provenance.py` and `src/train.py` | No provenance tracking. This is explicited by `TODO 1` in those files. A good metric cannot be traced back to what produced it because nothing records the git commit, library versions, data identity etc.. | We need to implement the run manifest and write it at the start of the run. We can verify it works by checking the `tests/test_provenance.py` in the MLflow. |
| `src/train.py` | Again explicited by `TODO 2 to 7`. Parameters, metrics per epoch, artifacts are not logged. You can't easily compare runs and pick a model based on the results. | We need to log everything to MLflow. We can check the logs in the MLflow UI to verify it works. |
| `src/validate_data.py` | (`TODO` again)No validation is performed on the data before training. This can lead to unexpected behavior or errors during training. | We need to implement data validation checks in `src/validate_data.py` to ensure the data is in the expected format and within acceptable ranges. |

#### Engineering checkpoint 1.

`model.pt` is the output of an experiment it stores weights, class_names, image_size and dropout but nothing about the what commit produced it, the hyperparameters used to train it, the data it was trained on etc...

### Milestone 1 - Make training repeatable

I am tasked to explain a few files before I begin:
- `pyproject.toml` is a written file that lists the project's direct dependencies and their versions. I say  versions but they are ranges so it doesn't guarantee that every collaborator will have the same version of the dependencies. 
- `uv.lock` is a generated file that completes the `pyproject.toml` by listing the exact versions of ALL dependencies and their transitive dependencies. This one is important for reproducibility.
- `.venv` is a folder that contains the virtual environment for the project. It is not tracked by git and should be ignored. It is important to have a virtual environment to isolate the project's dependencies from the system's Python installation especially when working on multiple projects with different dependencies.

#### Engineering checkpoint 2.

> Does `random_state=42` guarantee bit-for-bit identical results on every CPU/GPU/software stack?

Not really, a seed guarantees the same random numbers from the same generator. It doesn't guarantee the same arithmetic results. CUDA has it's own generator and kernels so whether you use a CPU or a GPU, the results may differ. Also, different versions of libraries may have different implementations of algorithms which can lead to different results even with the same seed. Software stack differences maybe also a factor.
