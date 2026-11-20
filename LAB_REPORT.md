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

### Milestone 2 - Extract and track the images

#### Engineering checkpoint 3.
> In one sentence each: what does the Git remote store, what does the DVC remote store, and does git push transfer the DVC-managed raw images?

- Git remote: It stores the versioned source code, configuration files (such as pyproject.toml, or the dvc metadata), and other text-based assets of the project.
- DVC remote: It stores the actual data files (like images, datasets, models) that are tracked by DVC, allowing for version control of large files without bloating the Git repository.
- Does git push transfer the DVC-managed raw images? No, `git push` only transfers the Git-tracked files to the remote repository. The DVC-managed raw images are stored in the DVC remote and need to be pushed separately using `dvc push`.

### Milestone 3 - Implement a Core data-quality gate

DONE

### Milestone 4 - Design the DVC pipeline from scratch


>Should PREPARE depend on reports/data_quality.json?

Yes, to my knowledge it is the only way DVC knows that PREPARE comes after VALIDATE.

>Should PREPARE also depend directly on data/raw?

Yes, PREPARE needs to go through the images in data/raw to prepare them for training. If the raw data changes, PREPARE should be re-run to ensure that the prepared data is up-to-date.

>Which changes in params.yaml should make TRAIN run again?

If the hyperparameters that affect the training process are changed, such as learning rate, batch size, number of epochs, or any other parameter that influences the model's training behavior, TRAIN should run again to reflect those changes in the training process.

>Why should EVALUATE track selection.run_id?

It tells use which model is being evaluated. If we don't track the run_id, we won't know which model's performance metrics are being reported in the evaluation phase, making it difficult to compare different models or runs.

>Which file contains scores that DVC should track as metrics?

`reports/test_metrics.json` contains the scores that DVC should track as metrics. This file typically includes evaluation metrics such as accuracy, precision, recall, F1-score, etc.....

### Milestone 5 - Record what each MLflow Run used

#### Engineering checkpoint 4.

>In one sentence per tool, explain what Git, DVC and MLflow each record about a training Run.

- Git records the version of the source code that led to the training run, including scripts, configuration files, and any other text-based assets.
- DVC records which data and stage outputs were used in the training run.
- MLflow records what happened during an execution of a training run, including parameters, metrics, artifacts, and the model itself. Runs can be compared and reproduced using the information logged by MLflow.

# Milestone 6 - Predict which stages need to run again

| Change | **VALIDATE** | **PREPARE** | **TRAIN** | **EVALUATE** |
| --- | --- | --- | --- | --- |
| Change `train.learning_rate` | REUSED | REUSED | RERUN | REUSED |
| Change `data.image_size` | REUSED | RERUN | RERUN | NEEDS UPDATE |
| Edit `src/validate_data.py` | RERUN | REUSED | REUSED | REUSED |
| Edit only `README.md` | REUSED | REUSED | REUSED | REUSED |

#### Engineering checkpoint 5.

Let's take a look at `data.image_size` change. `validate_data.py` never reads `params.yaml` and no parameter is declared so `VALIDATE` is `REUSED`. `PREPARE` as `image_size` in its parameters so it is `RERUN`. `TRAIN` also has `image_size` in its parameters and its dependencies get new hashes so it is `RERUN`. For `EVALUATE` it lists `image_size` and `data.processed/test` and the manifest changes so it is `NEEDS UPDATE`. The stage's outputs are not stale but the manifest is so it needs to be updated. Although using the same run id, it should mean that re-running `EVALUATE` would actually not work because the run's logged `image_size` would no longer match the new value I think, I'm not sure if it's an issue.

If we edit `src/validate_data.py`, `VALIDATE` is `RERUN` because the file is one of its dependencies and the file's hash changes. `PREPARE` is `REUSED` as `reports/data_quality.json` is one of its dependencies, the file is unchanged from VALIDATE even though the file got rewritten so its hash should be unchanged. DVC compares hashes not modifications times. `TRAIN` and `EVALUATE` are `REUSED` because nothing changed for them upstream.

#### Run and check the training stages

Given the type of model we deal with, I assume the main validation metric would be `val_loss` and perhaps `val_macro_f1` as well, using both would help me break a tie between one another. We want to see the loss decrease and the f1 increase over epochs.

If the learning rate is lower the weights will move slower, so finer updates and a slower convergence. So the loss could be higher using the same number of epochs. We trade stability for speed.

- Run ID (A) : cbbf401c41104b438bcadb025064e5d2
- Run ID (B) : 01642b2a11d54d078fc89a5c871de4a6 

#### Verify the learning rate change

| Check | **VALIDATE** | **PREPARE** | **TRAIN** | **EVALUATE** |
| --- | --- | --- | --- | --- |
| Baseline, Run A (first `dvc repro train`) | RERUN | RERUN | RERUN | NOT RUN |
| Re-run, nothing changed | REUSED | REUSED | REUSED | NOT RUN |
| Comment added to `src/validate_data.py` | RERUN | REUSED | REUSED | NOT RUN |
| `train.learning_rate = 0.0003`, Run B | REUSED | REUSED | RERUN | NOT RUN |

- `EVALUATE` is `NOT RUN` everywhere because dvc repro train only reproduces train and its upstream stages. `EVALUATE` is on another branch of the DAG.
- For the comment put in `src/validate_data.py`, it matches row 3 of the prediction table. `VALIDATE` reran because its code dependencies changed. It rewrote `data_quality.json` with identical bytes, so `PREPARE`'s dependency hash didn't change and DVC skipped it, and `TRAIN` with it. 
- For the learning-rate: it matches row 1. Only `TRAIN` lists train.learning_rate in its params.
- Baseline row: That's the only real difference kind of. Everything ran because `dvc.lock` didn't exist yet ("Generating lock file"), so DVC had no recorded state to compare with. 

| Run | lr | Final val_loss | Final val_acc | Final val_macro_f1 |
| --- | --- | --- | --- | --- |
| A | 0.001 | 0.8329 | 0.6391 | 0.6254 (peak 0.6445 at epoch 3) |
| B | 0.0003 | 0.8927 | 0.6165 | 0.6091 |

This matches what we predicted. The lower learning rate gives higher losses and lower scores after 4 epochs. B's curves are still clearly improving at epoch 4, so it's under-trained, not intrinsically worse. A's macro-F1 dips slightly at epoch 4 while its val_loss keeps decreasing.

### Milestone 7 - Compare three controlled runs

#### Create run C

After I changed back the learning rate to 0.001. I decided to change the batch size to 16 instead of 32. Our current diagnosis is that the model is under-trained, so a smaller batch size should give more updates per epoch and help it converge faster, meaning the loss should decrease and the scores should increase. I ran `dvc repro train` again to create run C.

- Run ID (C) : f6671d76abc44033ab5e59ac6ebf34f7

## Milestone 7 – Controlled comparison

All three Runs use the same prepared data (`training_data_identity` 8fa943ff…), seed (42), epochs (4), device (CPU) and locked environment. Scores are taken from the final epoch (epoch 4), because TRAIN logs the last-epoch model.

| Run | Controlled change | Val. Accuracy | Val. Macro-F1 | Val. loss | Training time | MLflow Run ID |
|-----|-------------------|---------------|---------------|-----------|---------------|---------------|
| A | Baseline (lr 0.001, batch 32, dropout 0.30, no augmentation) | 0.6391 | 0.6254 | 0.8329 | 10.90 s | `cbbf401c41104b438bcadb025064e5d2` |
| B | `train.learning_rate` 0.001 → 0.0003 | 0.6165 | 0.6091 | 0.8927 | 10.76 s | `01642b2a11d54d078fc89a5c871de4a6` |
| C | `train.batch_size` 32 → 16 and `learning_rate` 0.0003 → 0.001 | 0.6391 | 0.6349 | 0.8135 | 11.04 s | `f6671d76abc44033ab5e59ac6ebf34f7` |

### Engineering checkpoint 6

I chose validation macro-F1 as the primary metric because this is a three-class diagnosis problem in which every class matters equally. Here, failing to recognise one disease is a failure on us whatever the overall accuracy looks like. The classes are nearly balanced here (train 345/348/341, validation 44/45/44), so accuracy and macro-F1 move together, but they are not interchangeable. Runs A and C have exactly the same validation accuracy (85 of 133 images) and different macro-F1 (0.6254 vs 0.6349), which shows that their errors are distributed differently across the classes and that macro-F1 captures a difference accuracy cannot. Ties are broken by the lower final validation loss as stated above in milestone 6.

Models are selected using validation metrics rather than test scores to prevent optimistic bias on unseen data, and the pipeline enforces this by keeping the test set out of training and evaluating only a single frozen `selection.run_id` without depending on the training stage.

### Milestone 8 - Select and test on model

Looking at the table made in milestone 7, we can clearly see that run C has the best validation metrics without even a need for a tie-breaker. So I choose run C as the best model and set `selection.run_id` to `f6671d76abc44033ab5e59ac6ebf34f7`. 

On the 128 test images, the selected model reached a test accuracy of 0.7031 (90 correct) and a test macro-F1 of 0.6973. These are about six points higher than its validation scores (0.6391 and 0.6349). With 128 test and 133 validation images, one image is worth about 0.8 percentage points, and I believe a gap of this size (roughly eight images) is within the sampling variation expected from two small, independent splits.

The overall accuracy shown in the confusion matrix masks severe class imbalance in errors, especially bean rust recall at 0.51 (22 of 43) with 12 bean-rust leaves misclassified as healthy and leaving diseased crops untreated, which is why macro-F1 is the primary metric and why longer training and post-underfitting augmentation should be tested on validation before touching the test set again.

