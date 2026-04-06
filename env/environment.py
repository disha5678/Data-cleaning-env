from env.data_generator import generate_clean_dataset
from env.issue_injector import inject_issues
from env.actions import drop_correlated_feature, fill_nulls, remove_nulls, convert_types, deduplicate, trim_whitespace, normalize_column
from env.graders.task1_grader import grade_task1
from env.graders.task2_grader import grade_task2
from env.graders.task3_grader import grade_task3
from env.graders.final_evaluator import compute_final_score
class DataCleaningEnv:

    def __init__(self,task=1):
        self.task=task
        self.clean_df = None
        self.dirty_df = None
        self.manifest = None
        self.steps = 0
        self.done=False
        self.inspected_cols=set()

    def safe_df(self,df):
         return df.replace({float("nan"): None})
    
    def reset(self):
        self.clean_df = generate_clean_dataset()
        self.dirty_df, self.manifest = inject_issues(self.clean_df)

        self.steps = 0
        self.done = False
        self.inspected_cols = set()

        return {
            "dataset": self.safe_df(self.dirty_df).to_dict(),
            "shape": list(self.dirty_df.shape),
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

        if action_type == "remove_nulls":
            col = action["column"]
            null_ratio = self.dirty_df[col].isnull().mean()
            if null_ratio > 0.3:
                self.dirty_df = remove_nulls(self.dirty_df, col)
                reward += 0.1   # good decision
            else:
                reward -= 0.08  # bad decision 

        elif action_type == "convert_types":
            self.dirty_df = convert_types(self.dirty_df, action["column"])
            reward += 0.1

        elif action_type == "deduplicate":
            self.dirty_df = deduplicate(self.dirty_df)
            reward += 0.1

        elif action_type == "trim_whitespace":
            self.dirty_df = trim_whitespace(self.dirty_df, action["column"])
            reward += 0.1
        
        elif action_type == "fill_nulls":
            col = action["column"]
            null_ratio = self.dirty_df[col].isnull().mean()
            if null_ratio < 0.3:
                self.dirty_df = fill_nulls(self.dirty_df, col)
                reward += 0.12  # good decision
            else:
                reward -= 0.05

        elif action_type == "normalize":
             col = action["column"]

             if self.dirty_df[col].dtype != "object":
                self.dirty_df = normalize_column(self.dirty_df, col)
                reward += 0.1
             else:
                reward -= 0.08  # wrong column type    
        elif action_type == "drop_correlated":
            self.dirty_df = drop_correlated_feature(self.dirty_df, action["column"])
            reward += 0.1
        else:
            reward -= 0.05
            
        return {
            "dataset": self.safe_df(self.dirty_df).to_dict(),
            "shape":  list(self.dirty_df.shape),
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

        if self.task == 1:
            quality = grade_task1(agent_df, self.clean_df, self.manifest)

        elif self.task == 2:
            quality = grade_task2(agent_df, self.clean_df)

        elif self.task == 3:
            quality = grade_task3(agent_df, self.clean_df)

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