from torch.utils.tensorboard import SummaryWriter, FileWriter
from torch.utils.tensorboard.summary import scalar
import time
import torch

def log_single_search(writer, info, i, n_steps, dist_=None, pr=None):
    if info["T"] is not None:
        writer.add_scalar(f"T_{i}", info["T"], n_steps)  # T may be None
    if info["acceptance_prob"] is not None:
        writer.add_scalar(f"acceptance_prob_{i}", info["acceptance_prob"], n_steps)
    writer.add_scalar(f"num_seen_solutions_{i}", info["num_seen_solutions"], n_steps)
    writer.add_scalar(f"num_wasted_actions_{i}", info["num_wasted_actions"], n_steps)
    writer.add_scalars(f"distance_{i}", {"distance": info["distance"], "min_distance": info["min_distance"]}, n_steps)
    sum_action_counter = 1  # sum(info["action_counter"].values())
    writer.add_scalars(f"action_counter_{i}",
                       {op_name: count / sum_action_counter for op_name, count in info["action_counter"].items()},
                       n_steps)
    if dist_ is not None:
        writer.add_scalars(f"action_probs_{i}",
                           {op_name: val for op_name, val in zip(info["action_counter"].keys(), dist_[0])}, n_steps)
    if pr is not None:
        writer.add_scalars(f"action_probs_{i}",
                           {op_name: val for op_name, val in zip(info["action_counter"].keys(), pr)}, n_steps)

    if info['extra_params_to_log'] is not None:
        for p in info['extra_params_to_log']:
            writer.add_scalar(p+"_{}".format(i), info['extra_params_to_log'][p], n_steps)

def log_training(writer, info, i, score=None, baseline_scores=None, instance=None):
    if info["T_start"] is not None:
        writer.add_scalar("T_start", info["T_start"], i)
    if info["cs"] is not None:
        writer.add_scalar("cs", info["cs"], i)
    if info["len_d_E_history"] is not None:
        writer.add_scalar("len_d_E_history", info["len_d_E_history"], i)  # hopefully this is not 0!
    writer.add_scalar("num_seen_solutions", info["num_seen_solutions"], i)
    writer.add_scalar("min_distance", info["min_distance"], i)
    writer.add_scalar("num_wasted_actions", info["num_wasted_actions"], i)
    writer.add_scalar("num_best_improvements", info["num_best_improvements"], i)
    writer.add_scalar("num_improvements", info["num_improvements"], i)
    writer.add_scalars("action_counter", info["action_counter"], i)
    writer.add_scalar("min_step", info["min_step"], i)
    if info["warmup_phase_end"] is not None:
        writer.add_scalar("warmup_phase_end", info["warmup_phase_end"], i)
    if score is not None:
        writer.add_scalar("reward", score, i)
    if baseline_scores is not None and (
            i < len(baseline_scores) or (instance is not None and instance < len(baseline_scores))):
        if instance is not None:
            writer.add_scalar("improvement_over_baseline (%)",
                              100 * (baseline_scores[instance] - info["min_distance"]) / baseline_scores[instance], i)
            writer.add_scalars(f"min_distance_with_baseline", {"min_distance": info["min_distance"],
                                                               "baseline_min_distance": baseline_scores[instance]}, i)
        else:
            writer.add_scalar("improvement_over_baseline (%)",
                              100 * (baseline_scores[i] - info["min_distance"]) / baseline_scores[i], i)
            writer.add_scalars(f"min_distance_with_baseline",
                               {"min_distance": info["min_distance"], "baseline_min_distance": baseline_scores[i]}, i)


class mywriter(SummaryWriter):
    def __init__(self, log_dir):
        super(mywriter, self).__init__(log_dir)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        """Adds many scalar data to summary.

        Args:
            main_tag (string): The parent name for the tags
            tag_scalar_dict (dict): Key-value pair storing the tag and corresponding values
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            r = 5
            for i in range(100):
                writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                                'xcosx':i*np.cos(i/r),
                                                'tanx': np.tan(i/r)}, i)
            writer.close()
            # This call adds three values to the same scalar plot with the tag
            # 'run_14h' in TensorBoard's scalar section.

        Expected result:

        .. image:: _static/img/tensorboard/add_scalars.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_scalars")
        walltime = time.time() if walltime is None else walltime
        fw_logdir = self._get_file_writer().get_logdir()
        for tag, scalar_value in tag_scalar_dict.items():
            fw_tag = fw_logdir + "/" + main_tag.replace("/", "_") + "_" + tag
            assert self.all_writers is not None
            if fw_tag in self.all_writers.keys():
                fw = self.all_writers[fw_tag]
            else:
                fw = FileWriter(fw_tag, self.max_queue, self.flush_secs,
                                self.filename_suffix)
                self.all_writers[fw_tag] = fw
            if self._check_caffe2_blob(scalar_value):
                from caffe2.python import workspace
                scalar_value = workspace.FetchBlob(scalar_value)
            fw.add_summary(scalar(main_tag, scalar_value),
                           global_step, walltime)