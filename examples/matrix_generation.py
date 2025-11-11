from scope.compression import CompressionMatrix

compressor_names=['gzip']
compression_metric_names=['ncc']

# Setup
cm = CompressionMatrix(
    compressor_names=compressor_names,
    compression_metric_names=compression_metric_names,
    join_string=' ',
    n_jobs=2
)

# test_samples = {
#     1: [
#         "Una obra maestra cinematográfica que combina actuaciones excepcionales con una dirección brillante. Cada escena está cuidadosamente crafteada y la banda sonora es simplemente espectacular.",
#         "Increíble película que supera todas las expectativas. Los efectos visuales son impresionantes y la historia te mantiene en el borde del asiento desde el primer minuto hasta el último.",
#         "Una experiencia cinematográfica única e inolvidable. El guión es inteligente, los personajes están perfectamente desarrollados y la cinematografía es absolutamente hermosa.",
#     ],
#     0: [
#         "Una obra maestra cinematográfica que combina actuaciones excepcionales con una dirección brillante. Cada escena está cuidadosamente crafteada y la banda sonora es simplemente espectacular.",
#         "Una completa pérdida de tiempo que no logra conectar con la audiencia. El guión es predecible, las actuaciones son forzadas y la dirección carece de visión clara.",
#         "Película decepcionante que desperdicia un gran potencial. Los diálogos son torpes, la trama tiene agujeros enormes y los efectos especiales parecen de bajo presupuesto.",
#     ]
# }

# test_sample = "Fantástica película que combina acción emocionante con momentos de gran profundidad emocional. Los actores entregan performances convincentes y la dirección mantiene un ritmo perfecto throughout."

test_samples = {
    0: ['hola', 'alo', 'loa'],
    1: ['adios', 'bye', 'adiou']
}

test_sample = 'x'

# Test
print("Testing CompressionMatrix Serie...")

results = cm(samples=test_sample, kw_samples=test_samples)

print("Done!")