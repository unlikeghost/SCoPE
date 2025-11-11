from typing import List
from collections import OrderedDict
from scope.utils import make_report
from scope.optimization import ScOPEOptimizerAuto
from scope.optimization import ParameterSpace, all_subsets
from scope.compression.compressors import CompressorType
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


x_validation = [
    "molecule toxic heavy metal lead", "compound dangerous poison arsenic",
    "chemical harmful mercury substance", "element toxic cadmium dangerous",
    "poison lethal cyanide compound", "toxic substance benzene harmful",
    "dangerous chemical formaldehyde", "harmful compound asbestos fiber",
    "toxic metal chromium dangerous", "poison substance strychnine lethal",
    "harmful chemical dioxin toxic", "dangerous compound pesticide toxic",
    "safe molecule water oxygen", "harmless compound sugar glucose",
    "beneficial substance vitamin C", "safe chemical sodium chloride",
    "harmless element calcium safe", "beneficial compound protein amino",
    "safe substance carbohydrate energy", "harmless chemical citric acid",
    "beneficial molecule antioxidant", "safe compound fiber cellulose",
    "harmless substance mineral zinc", "beneficial chemical enzyme natural"
]

y_validation = [0]*12 + [1]*12

kw_samples_validation = [
    {
        0: ["toxic harmful dangerous poison lethal", "mercury lead arsenic cyanide"],
        1: ["safe harmless beneficial healthy natural", "water vitamin protein calcium"]
    }
    for _ in range(24)
]

x_train = [
    "chemical toxic pesticide harmful", "dangerous metal arsenic lead",
    "poisonous substance cyanide lethal", "harmful compound mercury cadmium",
    "toxic gas formaldehyde dangerous", "dangerous substance dioxin poison",
    "safe molecule water glucose", "beneficial compound vitamin protein",
    "harmless chemical citric acid", "safe element calcium magnesium",
    "safe substance carbohydrate fiber", "beneficial molecule antioxidant enzyme"
]

y_train = [0]*6 + [1]*6

kw_samples_train = [
    {
        0: ["toxic harmful dangerous poison lethal", "mercury lead arsenic cyanide"],
        1: ["safe harmless beneficial healthy natural", "water vitamin protein calcium"]
    }
    for _ in range(12)
]


auto_optimizer = ScOPEOptimizerAuto(
    random_seed=42,
    n_trials=10,
    target_metric='log_loss',
    study_name="test",
    n_jobs=1,
)

auto_study = auto_optimizer.optimize(x_train, y_train, kw_samples_train)

best_model = auto_optimizer.get_best_model()
auto_optimizer.save_complete_analysis()

all_y_true = y_validation
all_y_predicted = []
all_y_proba = []

pred: List[dict] = best_model(
    samples=x_validation,
    kw_samples=kw_samples_validation
)

for pred_x in pred:
    prediction: dict = pred_x['proba']
    sorted_dict = OrderedDict(sorted(prediction.items()))
    pred_key = max(sorted_dict, key=sorted_dict.get)
    predicted_class = int(pred_key.replace("sample_", ""))
    all_y_predicted.append(predicted_class)
    all_y_proba.append(list(sorted_dict.values()))

results = make_report(
    y_true=all_y_true,
    y_pred=all_y_predicted,
    y_pred_proba=all_y_proba,
    save_path='results'
)

print(f"AutoSampler best: {auto_study.best_value:.4f}")
print(results)