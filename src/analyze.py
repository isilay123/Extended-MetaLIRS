import pandas as pd

DEFAULT_ANALYZE_CSV  = './RESULTS/last_result.csv'
DEFAULT_ANALYZE_XLSX = './RESULTS/last_result.xlsx'
DEFAULT_CHECKPOINT_XLSX = './RESULTS/last_checkpoint.xlsx'
DEFAULT_EVAL_P_XLSX = './RESULTS/eval_p.xlsx'


EXAMPLE_ANALYZE_CSV = 'EXAMPLE_CSV.TXT'

# https://dataindependent.com/pandas/pandas-rank-rank-your-data-pd-df-rank/


def run_ranking(csv_file=DEFAULT_ANALYZE_CSV):
    data = pd.read_csv(csv_file)
    print(data)


if __name__ == "__main__":
    run_ranking(csv_file=EXAMPLE_ANALYZE_CSV)
