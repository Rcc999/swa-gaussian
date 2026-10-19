How to run tasks on Euler
=========================

This guide describes how you can run tasks on the ETH Euler cluster.
Please only follow this approach if you are unable to run the task on your local machine,
e.g., if you have very old hardware.
All the tasks are designed so that they can be completed on a personal laptop,
including if you are trying to obtain a competitive leaderboard score.
Almost all students will have a strictly smoother experience working
on their own laptop rather than following this guide.

**Read the "Important information" section very carefully before you start with this guide.
Failure to do so may result in loss of cluster access and/or termination of your ETH account!**

Note that you can adapt this guide to run the tasks on other systems, such as Google Colab. However, we can not provide any additional guidance or support for other approaches.

Important information
---------------------

1. **Never** perform **any computations** on Euler directly; **always use the batch system (sbatch)**! If you run your solution or other heavy computation directly, you will lose access to the cluster (possibly forever).
2. Please use this approach only as a last resort to not overload the cluster.
3. This is an unofficial approach, hence we can only provide very limited support to you.
4. The [ETH Zurich HPC documentation](https://docs.hpc.ethz.ch/) provides detailed documentation for the cluster, including an [FAQ](https://docs.hpc.ethz.ch/support/faq/). Whenever you have cluster-related questions, please search the documentation first. We will not answer questions that are already answered there.

Initial one-time setup
----------------------

The following steps prepare the cluster for running the tasks. You need to do those steps only once for the entire course.

1. Read *and understand* the documentation of the cluster and the rules for using it:
   1. Read [Getting started](https://docs.hpc.ethz.ch/tutorials/getting-started/) and [SSH into Euler](https://docs.hpc.ethz.ch/connections/ssh/), and follow the instructions to gain access. Connect to the ETH VPN first if you are outside the ETH network.
   2. Read [Your first job](https://docs.hpc.ethz.ch/tutorials/first-job/), especially how to submit a job, request resources, and monitor its progress.
   3. Revisit the [ETH Zurich Acceptable Use Policy for Information and Communications Technology (&#34;BOT&#34;)](https://rechtssammlung.sp.ethz.ch/Dokumente/203.21en.pdf).
2. Connect to the cluster and change into your home directory (`cd ~`) if necessary.
3. Download and install [uv](https://docs.astral.sh/uv/getting-started/installation/) (our Python package and project manager):
   1. Run `curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh`.
   2. Run `uv --version` and make sure that your uv version is *0.9.5*. If this does not work, try to disconnect and reconnect to the cluster.

Per-task setup
--------------

You need to perform the following steps only once for this task, but again for future tasks.

1. Upload the extracted handout to the cluster and store it as *~/task{x}/* ({x} is the number of the task). Your *solution.py* file should be stored as *~/task{x}/solution.py*. For that, use the following command:  `scp -r handout_folder_name user@euler.ethz.ch:~/`, where you should insert the name of the handout folder and your ETH username in the "user" field in the email address. This will copy the handout folder to the cluster.
2. Connect to the cluster and change into the task directory by running `cd ~/task{x}/`.
3. Create the task's Python environment:
   1. Run `uv venv --python 3.8 .venv`.
   2. Run `uv run python --version` and make sure that your Python version starts with *3.8*.
   3. Run `uv pip install --no-cache -r requirements.txt`
   4. Whenever you change *requirements.txt*, do the following:
   5. Always run commands via `uv run COMMAND` to make sure that they use the task's Python environment!
   6. If you added a new package, you need to re-run `uv pip install --no-cache -r requirements.txt`.
   7. If you removed a package, you need to run `uv pip uninstall PACKAGE` where `PACKAGE` is the package name.
4. After finishing task {x} you can free some space by removing the environment via `rm -rf .venv`.

Running your code
-----------------

You need to perform the following steps each time you reconnect to the cluster and want to run your solution.

**Only run your solution via the batch system, and never directly! If you run your solution directly, you will lose access to the cluster now and in the future!**

Please follow the guide below carefully.

To submit a job to the batch system, run the following command from *~/task{x}/*:

`sbatch -n 4 --mem-per-cpu=2048 --wrap="uv run python -u checker_client.py --results-dir ."`

1. Do **not** modify any files in the *~/task{x}/* directory until the batch job has completed. The cluster uses your code at the time of *execution*, not at the time of *submission*!
2. You can inspect the state of the all pending, running and shortly finished jobs by running `squeue`.
3. After your job starts, you will obtain its job ID via console and you can use `scontrol show jobid -dd jobID` (where your replace `jobID` with the job's ID) to display information about it.
4. As soon as your job is completed, you can find its console output in *~/task{x}/slurm-jobID.out* (*jobID* is the ID of your job) and your submission in *~/task{x}/results_check.byte*.

After successful run and obtaining of the *results_check.byte* file, open terminal on your local machine and run `scp -r user@euler.ethz.ch:~/task{x}/results_check.byte /local/dir/`, where */local/dir* denotes the directory where you want your results file stored. Finally, you can upload this results file in the submission system.
