run('/home/ruben/Documentos/trunk-eit-dataset/src/forward_process/scripts/startup_eidors.m');
addpath('/home/ruben/Documentos/trunk-eit-dataset/src/forward_process/scripts');
startup_eidors();
run_forward(
    16,
    '/tmp/tmpiqiq3l0v/grid.txt',
    -1.0, 1.0,
    -1.0, 1.0,
    1.0,
    '/tmp/tmpiqiq3l0v'
);