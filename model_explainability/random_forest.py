from base_logger import log
log = log.getLogger(__name__)

import numpy as np
import pandas as pd


def plot_activation(classifier, record):
    #plt.imshow(record.reshape(28, 28))
    #y, x = divmod(np.array(path_features), 28)
    #plt.scatter(x, y, c = 'red', alpha = 1 - np.array(path_probability), s = 100, marker = 'o')
    #plt.close()
    pass


def decission_nodes(classifier, record):
    if getattr(classifier, 'feature_names_in_', None) is None:
        feature_names = np.array(range(record.shape[1]))
    else:
        feature_names = classifier.feature_names_in_
        record = record.values

    decission_nodes = []
    for estimator in classifier.estimators_:
        prediction   = estimator.predict(record)[0]
        feature_idx  = estimator.tree_.feature

        assert record.shape[1] == estimator.tree_.n_features

        path = estimator.decision_path(record).indices

        path_operators = [
            "<=" if estimator.tree_.children_left[node] == child else ">"
            for node, child in zip(path[:-1], path[1:])
        ] 

        #for the moment ignore the fact that these probabilities are probably conditional 
        #TODO: more accurately calculate node entrpy with conditional probabilities on parent nodes
        path_probability = [
            estimator.tree_.n_node_samples[child] / estimator.tree_.n_node_samples[node]
            for node, child in zip(path[:-1], path[1:])
        ]  

        path_thresholds = [estimator.tree_.threshold[node] for node in path[:-1]] #last node does not split data
        path_features = [feature_names[feature_idx[node]] for node in path[:-1]]

        decission_nodes.append(
            pd.DataFrame({
                'feature'            : path_features,
                'encoded_value'      : record[0, path_features],
                'operator'           : path_operators,
                'threshold'          : path_thresholds,
                'predicted_class'    : prediction,
                'sample_probability' : path_probability
            })
        )

    decission_nodes = pd.concat(decission_nodes)

    return decission_nodes 
    

def explain(classifier, record):
    #record = image
    assert len(record) == 1, f"Can only explain output for one record at a time, but got {len(record)}"

    output_class = classifier.predict(record)[0]
    probability  = classifier.predict_proba(record)[0, output_class == classifier.classes_][0]

    log.info(f"Output class {output_class} with probability: {probability}")

    is_bool_encoded = pd.Series([type(y) is np.bool_ for y in record[0]])

    nodes = (
        decission_nodes(classifier, record)
        .eval('is_bool_encoded = feature.map(@is_bool_encoded)')
    )

    # Aggregating the entropy of category nodes as a gross simplification
    # lets say x is in [A, B, C, D] as values 
    # lets say x == A
    # a node asses that x != B based on it's samples
    # what is the node's P(x = A | x != B) over the samples so that we can calculate entropy
    # From bayes theory P( x = A | x != B) = P(x != B | x = A) * P(x = A) / P(x != B)
    # P( x != B | x = A) is always 1 thogh, because if x = A then x != B deffinitely
    # => P( x = A | x != B) = P(x = A) / P(x != B)
    # P(x = A) is not known from the samples, but we could use the training set probability instead
    # to ensure the probabilities are not > 1 we have to use P(x != B) from training set too
    # => P( x = A | x != B) = P(x = A) / (1 - P(x = B))

    log.info("Aggregate entropy for numeric nodes")
    numeric_values = (
        nodes
        .query('is_bool_encoded == False')
        .rename( columns = { 'threshold' : 'value' })
        .assign(
            sample_entropy = lambda x: -np.log2(x['sample_probability']),
            count = 1
        )
    )

    def sort_sum_filter(grp, pct_entropy):
        ascending_order = False if '>' in grp['operator'].values else True
        value_selection = 'right' if ascending_order else 'left'

        entropy_levels = (
            grp
            .sort_values('value', ascending = ascending_order)
            .assign( 
                sample_entropy = lambda x: x['sample_entropy'].cumsum(),
                pct_entropy    = lambda x: 100 * x['sample_entropy'] / x['sample_entropy'].iloc[-1],
                count          = lambda x: x['count'].cumsum(),
            )
        )

        total_entropy = (
            entropy_levels
            .query('pct_entropy == 100')
            .assign(
                value = entropy_levels.query('pct_entropy > @pct_entropy')['value'].values[0]
            )
        )
        
        return total_entropy

    numeric_entropy = []
    for col in numeric_values['feature'].unique():
        #col = 'Age'
        col_values = numeric_values.query('feature == @col')
        #NOTE: not sure this is the right implementation for Freedman Diaconis rule
        Q1 = col_values['value'].quantile(0.25)
        Q3 = col_values['value'].quantile(0.75)
        IQR = Q3 - Q1
        bins =  round(2 * IQR / col_values.shape[0] ** (1/3))

        numeric_entropy.append(
            col_values
            .groupby(['feature', 'operator', 'predicted_class'])
            [col_values.columns]
            .apply(
                lambda grp: sort_sum_filter(grp, pct_entropy = 80),
                include_groups = False
            )
            .reset_index(drop = True)
            #.sort_values(['predicted_class', 'value'])
        )
       
    numeric_entropy = pd.concat(numeric_entropy)

    return numeric_entropy[['feature', 'predicted_class', 'operator', 'value', 'sample_entropy']] 
    #(
    #    numeric_entropy
    #    #.query('feature == 350')
    #    .sort_values(['feature', 'value'])
    #    .groupby(['feature', 'predicted_class'])
    #    .agg({
    #        'sample_entropy' : 'sum',
    #        'value' : lambda x: x.tolist(),
    #        'operator' : lambda x: x.tolist()
    #    })
    #)
