run('/home/ruben/Documentos/trunk-eit-dataset/src/forward_process/scripts/startup_eidors.m');
addpath('/home/ruben/Documentos/trunk-eit-dataset/src/forward_process/scripts');
startup_eidors();
run_forward(
    16,
    '/tmp/tmp5a7_1c76/grid.txt',
    -1.019687764452461, 1.019687764452461,
    -1.019687764452461, 1.019687764452461,
    0.9888684138432338,
    '/tmp/tmp5a7_1c76'
);