from .imputer_base import ImputerBase


# IMPORTS HERE


IMPUTE_NAME = "IMPUTATION_STRATEGY_NAME"


#
# COMMENTS
# LINKS TO DOCS
#


class Imputer(ImputerBase):

    def __init__(self):
        super().__init__(IMPUTE_NAME)

    def impute(self, df, column_names=None):
        raise Exception('INCOMPLETE, IMPUTATION CODE HERE')


#
# Run this template as "python -m imputers.template"
#
if __name__ == "__main__":
    Imputer().random_test()
