from scope.model import SCoPE

compressor_names=['lz77', 'rle']
compression_metric_names=['ncc', 'ncd']
distance_metrics=['euclidean']
prototype_method=None


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

model = SCoPE(
    distance_metrics=distance_metrics,
    prototype_method=prototype_method,
    compressor_names=compressor_names,
    compression_metric_names=compression_metric_names,
    join_string=' ',
)


prediction = model(
    kw_samples=test_samples,
    samples=test_sample,
)

print(prediction)