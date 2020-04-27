from arm_pan17_stylebreach_evaluator import main

train_path = '/Users/hekpo/PycharmProjects/Karas/for_arm/data/train_dataset'
test_path = '/Users/hekpo/PycharmProjects/Karas/for_arm/data/test_dataset'
train_results_path = '/Users/hekpo/PycharmProjects/Karas/for_arm/results/train_results'
test_results_path = '/Users/hekpo/PycharmProjects/Karas/for_arm/results/test_results'
train_output_path = '/Users/hekpo/PycharmProjects/Karas/for_arm/results/train_results'
test_output_path = '/Users/hekpo/PycharmProjects/Karas/for_arm/results/test_results'

"""
from eng_pan17_stylebreach_evaluator import main

train_path = '/Users/hekpo/PycharmProjects/Karas/for_eng/data/pan17-style-breach-detection-training-dataset-2017-02-15'
test_path = '/Users/hekpo/PycharmProjects/Karas/for_eng/data/pan17-style-breach-detection-test-dataset-2017-02-15'
train_results_path = '/Users/hekpo/PycharmProjects/Karas/for_eng/results/train_results'
test_results_path = '/Users/hekpo/PycharmProjects/Karas/for_eng/results/test_results'
"""

main(train_path,train_results_path,train_results_path)
#main(test_path,test_results_path,test_results_path)
