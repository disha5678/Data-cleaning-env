from env.data_generator import generate_clean_dataset
from env.issue_injector import inject_issues
from env.actions import remove_nulls, convert_types, deduplicate
class DataCleaningEnv:

    def __init__(self):
        self.clean_df = None
        self.dirty_df = None
        self.manifest = None
        self.steps = 0

    def reset(self):
        self.clean_df = generate_clean_dataset()
        self.dirty_df, self.manifest = inject_issues(self.clean_df)
        self.steps = 0

        return self.dirty_df

    def step(self, action):
        self.steps += 1
        if action["type"] == "remove_nulls":
            self.dirty_df = remove_nulls(self.dirty_df, action["column"])
        
        return self.dirty_df, 0, False, {}

    def submit_cleaned_data(self, df):
        return df