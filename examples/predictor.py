from scope.compression import CompressionMatrix
from scope.predictor import  SCoPEPredictorV1

compressor_names=['lz77', 'rle']
compression_metric_names=['ncc', 'ncd']

cm = CompressionMatrix(
    compressor_names=compressor_names,
    compression_metric_names=compression_metric_names,
    join_string=' ',
    n_jobs=2
)

test_samples = {
    0: [
        "Increíble película que supera todas las expectativas. Los efectos visuales son impresionantes y la historia te mantiene en el borde del asiento desde el primer minuto hasta el último.",
        "Una experiencia cinematográfica única e inolvidable. El guión es inteligente, los personajes están perfectamente desarrollados y la cinematografía es absolutamente hermosa.",
    ],
    1: [
        "Una completa pérdida de tiempo que no logra conectar con la audiencia. El guión es predecible, las actuaciones son forzadas y la dirección carece de visión clara.",
        "Película decepcionante que desperdicia un gran potencial. Los diálogos son torpes, la trama tiene agujeros enormes y los efectos especiales parecen de bajo presupuesto.",
    ]
}

test_sample = "Fantástica película que combina acción emocionante con momentos de gran profundidad emocional."


compression_matrix = cm(samples=test_sample, kw_samples=test_samples)


predictor = SCoPEPredictorV1(
    evaluation_metrics=['wasserstein']
)

result = predictor(
    list_of_cm=compression_matrix
)

print(result)