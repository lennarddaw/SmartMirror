Model A: Rep segmentation
- already done, TCN model

Model B: Exercise classification
- build a binary good/bad form classifier for the respective exercise (e.g. with RandomForestClassifier)
- use model.feature:importances_ to identify problem joints (builds binary trees and finds out which joints separate the classes most optimally)
- convert this information to user feedback

Model C: Posture feedback
- multi-class classification of exercises (e.g. RandomForestClassifier)

RandomForestClassifier example (literally one line):
model_form = RandomForestClassifier().fit(X, form_labels)

Roadmap:
- create model B
- train model A on multiple exercises
- create B-A-pipeline
- create model C
- extend pipeline by model C
- develop frontend application
- build mirror