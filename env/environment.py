from env.data_generator import generate_clean_dataset
from env.issue_injector import inject_issues
from env.actions import remove_nulls, convert_types, deduplicate
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
            "dataset": self.dirty_df,
            "shape": self.dirty_df.shape
        }

    def step(self, action):
        self.steps += 1
        reward = 0

        # placeholder (integration later)
        if action["type"] == "inspect_column":
            col = action["column"]
            if col not in self.inspected_cols:
                self.inspected_cols.add(col)
                reward += 0.01
            else:
                reward -= 0.02

        return {
            "dataset": self.dirty_df
        }, reward, self.done, {}

    def state(self):
        return {
            "steps": self.steps,
            "dataset_shape": self.dirty_df.shape,
            "inspected_columns": list(self.inspected_cols)
        }

    def submit_cleaned_data(self, agent_df):
        self.done = True

        return {
            "agent_output": agent_df,
            "ground_truth": self.clean_df
        }

class StateManager:
    def __init__(self):
        self.steps = 0
        self.inspected_cols = set()