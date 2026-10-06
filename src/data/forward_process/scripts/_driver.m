run('/home/ruben/Documentos/trunk-eit-dataset/src/data/forward_process/scripts/startup_eidors.m');
addpath('/home/ruben/Documentos/trunk-eit-dataset/src/data/forward_process/scripts');
startup_eidors();
run_forward(
    16,
    'opposite',
    '/tmp/tmpacir3j70/grid.mat',
    -1.054191708149918, 1.054191708149918,
    -1.054191708149918, 1.054191708149918,
    1.0444842679342092,
    '/tmp/tmpacir3j70'
);