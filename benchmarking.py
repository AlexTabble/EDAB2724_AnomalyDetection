import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score

from great_tables import GT,md


class Benchmarking:
    """
    Class for benchmarking anomaly detection models
    """

    def __init__():
        pass

    @staticmethod
    def create_anomaly_groups(data : pd.DataFrame|pd.Series, col='outlier',
                              include_single_groups=False,
                              show_printout=True,
                              merge_tolerance=5,
                              noise_tolerance=3) -> list[tuple[int]]:
        """
        Creates list of tuples containing start and end indices of anomalous regions
        
        ---
        Parameters
        - data (datalike): Either a dataframe or series
        - col (str): Needs to be specified if a dataframe 
        - include_single_groups (boolean): whether to include anomalous regions
                                            which has length of 1
        - merge_tolerance (int) : threshold for gaps between anomalies and when
                                    its appropriate to merge them.
        - noise tolerance (int) : threshold for when groups are considered noise
                                    and not truly anomalous regions
        ---
        Output
         [(start_1,end_1),...,(start_n,end_n)]

        ---
        Example

        output = Benchmarking.create_anomaly_groups(data)
        print(output)

        >>> [(30,35),...,(8000,8029)]
        """
        
        group_ids = None
        # StackOverflow magic that creates cumsum of anomaly col
        if type(data) is pd.DataFrame:
            group_ids = data[col].ne(data[col].shift()).cumsum()
        else:
            group_ids = data.ne(data.shift()).cumsum()
            

        grouped = data.groupby(group_ids)

        groups = []

        for group_id, group in grouped:

            if type(group) is pd.DataFrame:
                if group[col].iloc[0] == False:
                    continue
            else:
                if group.iloc[0] == False:
                    continue
            
            # If a single instance is an anomaly, skip or not?
            if len(group) == 1 & include_single_groups: continue 
            
            indices = group.index.tolist()
            
            groups.append(
                (
                    indices[0],
                    indices[-1] + 1 # Last index is exclusive so increment by 1
                )
            )

        merged = []
        start_prev,end_prev = groups[0][0], groups[0][1]
        for idx, _ in enumerate(groups):
            
            if idx == 0: continue
            
            start_current = groups[idx][0]
            end_current = groups[idx][1]
            
            if (start_current - end_prev) <= merge_tolerance:
                end_prev = end_current
            else:
                merged.append((start_prev, end_prev))
                start_prev = start_current
                end_prev = end_current
        merged.append((start_prev, end_prev))
                
        groups = merged
        
        groups = [group for group in groups if (group[1] - group[0]) > noise_tolerance] 
            
            
        if show_printout:
            print(f'{len(groups)} anomaly groups identified')
        return groups

    @staticmethod
    def evaluate_model(y_true : np.array, y_pred : np.array) -> pd.DataFrame:
        """
        Returns a DataFrame which contains the metrics for the model

        ---
        Parameters
        - y_true (np.array) : True outlier series
        - y_pred (np.array) : Predicted outlier series

        ---
        Output
        pd.DataFrame

        |                   | Score  |
        |-------------------|--------|
        | Accuracy          | 20     |
        | Precision         | 40     |
        | Recall            | 89     |
        | Balanced Accuracy | 89     |
        | Groups Accuracy   | 40     |

        ---
        Example
        
        metrics = Benchmarking.evaluate_model(y_true,y_pred)
        print(metrics)
        
        """

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

        predicted_groups = Benchmarking.create_anomaly_groups(pd.Series(y_pred))
        true_groups = Benchmarking.create_anomaly_groups(pd.Series(y_true))

        group_accuracy = Benchmarking._evaluate_groups(predicted_groups, true_groups,
                                                       group_penalty=False)
        penalized_group_accuracy = Benchmarking._evaluate_groups(predicted_groups,
                                                                 true_groups)

        metrics = pd.DataFrame({
            'Score' : [
                round(accuracy*100,2),
                round(precision*100,2),
                round(recall*100,2),
                round(balanced_accuracy*100,2),
                round(group_accuracy*100,2),
                round(penalized_group_accuracy*100,2)
            ]
        }, index = ['Accuracy','Precision','Recall','Balanced Accuracy',
                    'Group Accuracy','Penalised Group Accuracy'])
        return metrics

    @staticmethod
    def _evaluate_groups(predicted_groups : list[tuple[int]],
                         true_groups : list[tuple[int]],
                         group_penalty : bool = True,
                         show_printout :bool = True) -> float:

        if show_printout:
            if len(predicted_groups) > 10:
                print(f'Model predicts {len(predicted_groups) -10} more than 10')
            elif len(predicted_groups) < 10:
                print(f'Model predicts {10 - len(predicted_groups)} less than 10')
            else:
                print('Number of groups match!')
                   
        actual_starts = [idx[0] for idx in true_groups]
        actual_ends = [idx[1] for idx in true_groups]

        valid_preds = 0
        bad_preds = []
        for pred in predicted_groups:
            is_start_correct = pred[0] in actual_starts
            is_end_correct = pred[1] in actual_ends
            
            if is_end_correct & is_end_correct : valid_preds +=1
            else:
                bad_preds.append(pred)

        accuracy = valid_preds / len(true_groups)

        # PERF: I'm not sure wether this is appropriate
        # This is on the assumption that the unseen data also contains
        # exactly 10 anomalies which means the model does not necessarly generalise
        # to unseen data where the anomaly count is known.
        # NOTE: I will ask whether this is appropriate
        # For testing purposes, I'm including this but use it by keeping data leakage
        # in mind
        if group_penalty:
            penalty = min(len(predicted_groups), len(true_groups)) / max(
                len(predicted_groups), len(true_groups)
            )
            accuracy = accuracy * penalty
        
        return accuracy
        

    @staticmethod
    def print_evaluation(y_true : np.array, y_pred : np.array, model_name: str) -> None:

        metrics = Benchmarking.evaluate_model(y_true, y_pred)
        metrics = metrics.reset_index().rename({'index' : 'Metric'})

        (
            GT(metrics)
            .tab_header(md(f'Model Results for **{model_name}**'))
            .tab_source_note(md("Metrics are in percentage(%)"))
        ).show()

