% Complete EIT simulation script
% Input: n_electrodes, base_conductivity, anomalies_json
% Output: Saves nodes, elems, voltages, elem_data to tmpdir
%
% Anomaly JSON format:
%   {
%     "shape": {"shape_type": "circle", "cx": 0.3, "cy": 0.2, "radius": 0.15, ...},
%     "conductivity": 0.5
%   }

function simulate(n_elec, base_cond, anomalies_json, tmpdir)
    if nargin < 2
        base_cond = 1.0;
    end
    if nargin < 3
        anomalies_json = '[]';
    end
    if nargin < 4
        tmpdir = '/tmp';
    end

    run('/home/ruben/Documentos/EIDORS_implementation/eidors_lib/eidors-v3.12/eidors/startup.m');

    imdl = mk_common_model('d2d1c', n_elec);
    img = mk_image(imdl, base_cond);

    nodes = img.fwd_model.nodes;
    elems = img.fwd_model.elems;
    n_elem = size(elems, 1);

    elem_data = ones(n_elem, 1) * base_cond;

    if ~strcmp(anomalies_json, '[]')
        anomalies = jsondecode(anomalies_json);

        for i = 1:n_elem
            n1 = elems(i, 1);
            n2 = elems(i, 2);
            n3 = elems(i, 3);

            cx = mean([nodes(n1, 1), nodes(n2, 1), nodes(n3, 1)]);
            cy = mean([nodes(n1, 2), nodes(n2, 2), nodes(n3, 2)]);

            for j = 1:numel(anomalies)
                a = anomalies(j);
                shape = a.shape;
                shape_type = shape.shape_type;

                inside = false;

                % Check based on shape type
                switch shape_type
                    case 'circle'
                        dist = sqrt((cx - shape.cx)^2 + (cy - shape.cy)^2);
                        inside = dist <= shape.radius;

                    case 'ellipse'
                        dx = cx - shape.cx;
                        dy = cy - shape.cy;
                        cos_r = cos(shape.rotation);
                        sin_r = sin(shape.rotation);
                        rx = (dx * cos_r + dy * sin_r) / shape.rx;
                        ry = (-dx * sin_r + dy * cos_r) / shape.ry;
                        inside = (rx^2 + ry^2) <= 1;

                    case 'harmonic'
                        r = sqrt((cx - shape.cx)^2 + (cy - shape.cy)^2);
                        theta = atan2(cy - shape.cy, cx - shape.cx);
                        r_shape = shape.base_radius;
                        for n = 1:shape.n_harmonics
                            a_n = shape.a_coeffs(n);
                            b_n = shape.b_coeffs(n);
                            r_shape = r_shape + a_n * cos(n * theta) + b_n * sin(n * theta);
                        end
                        inside = r <= r_shape;

                    case 'rectangle'
                        inside = (cx >= shape.x_min && cx <= shape.x_max && ...
                                  cy >= shape.y_min && cy <= shape.y_max);

                    otherwise
                        warning(['Unknown shape type: ', shape_type]);
                end

                if inside
                    elem_data(i) = a.conductivity;
                    break;
                end
            end
        end
    end

    img.elem_data = elem_data;
    vh = fwd_solve(img);

    dlmwrite(fullfile(tmpdir, 'nodes.txt'), nodes, ' ');
    dlmwrite(fullfile(tmpdir, 'elems.txt'), elems, ' ');
    dlmwrite(fullfile(tmpdir, 'voltages.txt'), vh.meas, ' ');
    dlmwrite(fullfile(tmpdir, 'elem_data.txt'), elem_data, ' ');
end