from env.data_generator import generate_clean_dataset
from env.issue_injector import inject_issues
from env.actions import remove_nulls, convert_types, deduplicate, trim_whitespace
from env.graders.task1_grader import grade_task1
from env.graders.final_evaluator import compute_final_score
class DataCleaningEnv:

    def __init__(self):
        self.clean_df = None
        self.dirty_df = None
        self.manifest = None
        self.steps = 0
        self.done=False
        self.inspected_cols=set()

    def reset(self):
        self.clean_df = generate_clean_dataset()
        self.dirty_df, self.manifest = inject_issues(self.clean_df)

        self.steps = 0
        self.done = False
        self.inspected_cols = set()

        return {
            "dataset": self.dirty_df.to_dict(),
            "shape": self.dirty_df.shape,
            "steps": self.steps
        }

    def step(self, action):
        self.steps += 1
        reward = 0
        action_type = action.get("type")

        if action_type == "inspect_column":
           col = action["column"]
           if col not in self.inspected_cols:
              self.inspected_cols.add(col)
              reward += 0.01
           else:
             reward -= 0.02

        elif action_type == "remove_nulls":
             col = action["column"]

             if col in self.manifest["nulls"]:
                self.dirty_df = remove_nulls(self.dirty_df, col)
                reward += 0.1
             else:
               reward -= 0.05

        elif action_type == "convert_types":
            self.dirty_df = convert_types(self.dirty_df, action["column"])
            reward += 0.1

        elif action_type == "deduplicate":
            self.dirty_df = deduplicate(self.dirty_df)
            reward += 0.1

        elif action_type == "trim_whitespace":
            self.dirty_df = trim_whitespace(self.dirty_df, action["column"])
            reward += 0.1

        else:
            reward -= 0.05
        return {
            "dataset": self.dirty_df.to_dict(),
            "shape": self.dirty_df.shape,
            "steps": self.steps
        }, reward, self.done, {}

    def state(self):
        return {
            "steps": self.steps,
            "dataset_shape": self.dirty_df.shape,
            "inspected_columns": list(self.inspected_cols)
        }

    def submit_cleaned_data(self, agent_df):
        self.done = True

        quality = grade_task1(agent_df, self.clean_df, self.manifest)
        final = compute_final_score(quality, self.steps)

        return {
            "quality_score": quality,
            "steps": self.steps,
            "final_score": final
        }

class StateManager:
    def __init__(self):
        self.steps = 0
        self.inspected_cols = set()