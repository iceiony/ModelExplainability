import tests
import pytest

from importlib import reload;
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from imblearn.metrics import geometric_mean_score
from sklearn.metrics import accuracy_score

import mnist as mn
import random_forest as rf


def test_can_extract_explanation_from_random_forest_type_models():
    images, labels = mn.load_mnist_data('./data')

    train, test, labels, target = train_test_split(images, labels, test_size=0.2)

    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(train.reshape(-1, 28 * 28), labels)

    train_prediction = classifier.predict(train.reshape(-1, 28 * 28))

    #assert model can fit the data
    assert geometric_mean_score(labels, train_prediction) > 0.9
    assert accuracy_score(labels, train_prediction) > 0.9

    test_prediction = classifier.predict(test.reshape(-1, 28 * 28))
    
    #assert model can predict on unseen data
    assert geometric_mean_score(target, test_prediction) > 0.9
    assert accuracy_score(target, test_prediction) > 0.9

    missclassified = np.where(target != test_prediction)[0]
    classified_correctly = np.where(target == test_prediction)[0]

    image = test[classified_correctly[0]].reshape(1, -1)
    label = target[classified_correctly[0]]
    reason = rf.explain(classifier, image)

    class_entropy = (
        reason
        .groupby('predicted_class')
        .agg({'sample_entropy' : 'sum'})
        .sort_values('sample_entropy', ascending = False)
        .assign(
            pct_info = lambda x: 100 * x['sample_entropy'] / x['sample_entropy'].sum()
        )
    )

    assert class_entropy.loc[label, 'sample_entropy'] == class_entropy['sample_entropy'].max(), \
        f'Expected {label} to have the highest entropy, but got label {class_entropy.iloc[0].name} instead'

    image = test[missclassified[0]].reshape(1, -1)
    label = target[missclassified[0]]
    predicted = test_prediction[missclassified[0]]

    reason = rf.explain(classifier, image)

    class_entropy = (
        reason
        .groupby('predicted_class')
        .agg({'sample_entropy' : 'sum'})
        .sort_values('sample_entropy', ascending = False)
        .assign(
            pct_info = lambda x: 100 * x['sample_entropy'] / x['sample_entropy'].sum()
        )
    )

    assert class_entropy.loc[predicted, 'sample_entropy'] == class_entropy['sample_entropy'].max(), \
        f'Expected {label} to have the highest entropy, but got label {class_entropy.iloc[0].name} instead'

    assert class_entropy.loc[label, 'sample_entropy'] < class_entropy.loc[predicted, 'sample_entropy'], \
        f'Expected {label} to have lower entropy than {predicted}, but got {class_entropy.loc[label, "sample_entropy"]} and {class_entropy.loc[predicted, "sample_entropy"]} instead'

    feature_entropy = (
        reason
        .reset_index(drop = False)
        .sort_values('sample_entropy', ascending = False)
        .assign(
            intensity = lambda x: np.abs(x['sample_entropy']) / np.abs(x['sample_entropy']).max(),

            colour = lambda x: np.where(
                x['operator'].apply(lambda y: y[0]) == '>', 
                'red', 'blue'),

            is_target_class = lambda x: x['predicted_class'] == label,  
        )
        .query('predicted_class == 3')
    )

    import matplotlib.pyplot as plt;plt.ion()

    
    plt.imshow(image.reshape(28, 28))
    plt.imshow(np.zeros([28, 28]))
    y, x = divmod(feature_entropy['feature'], 28)
    plt.scatter(
        x, y, 
        c = feature_entropy['colour'],
        alpha = feature_entropy['intensity'], 
        s = 100, marker = 's'
    )
    plt.close()
