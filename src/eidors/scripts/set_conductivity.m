% Set conductivity values for elements based on anomalies
% Input: img, anomalies (struct array with cx, cy, radius, conductivity)
% Output: img with updated elem_data

function img_out = set_conductivity(img, anomalies, base_conductivity)
    if nargin < 3
        base_conductivity = 1.0;
    end

    nodes = img.fwd_model.nodes;
    elems = img.fwd_model.elems;
    n_elem = size(elems, 1);

    elem_data = ones(n_elem, 1) * base_conductivity;

    if ~isempty(anomalies)
        for i = 1:n_elem
            n1 = elems(i, 1);
            n2 = elems(i, 2);
            n3 = elems(i, 3);

            cx = mean([nodes(n1, 1), nodes(n2, 1), nodes(n3, 1)]);
            cy = mean([nodes(n1, 2), nodes(n2, 2), nodes(n3, 2)]);

            for j = 1:numel(anomalies)
                a = anomalies(j);
                dist = sqrt((cx - a.cx)^2 + (cy - a.cy)^2);
                if dist <= a.radius
                    elem_data(i) = a.conductivity;
                    break;
                end
            end
        end
    end

    img_out = img;
    img_out.elem_data = elem_data;
end