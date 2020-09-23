let _points = [];
let _nx, _ny, _nz;

let x, y, z;
let sigma, beta, rho;
let stepSize;

function init(
  _sigma = 10.0,
  _beta = 2.667,
  _rho = 28.0,
  _x = 1,
  _y = 1,
  _z = 1,
  _stepSize = 0.004,
  _bailout = 1e10
) {
  _points = [];
  sigma = _sigma;
  beta = _beta;
  rho = _rho;
  x = _x;
  y = _y;
  z = _z;
  stepSize = _stepSize;
  bailout = _bailout;
}

function next(amount = 1) {
  while (amount > 0) {
    _nx = sigma * (y - x);
    _ny = x * (rho - z) - y;
    _nz = x * y - beta * z;

    x += stepSize * _nx;
    y += stepSize * _ny;
    z += stepSize * _nz;

    if (Math.abs(x) > bailout || Math.abs(y) > bailout || Math.abs(z) > bailout)
      break;

    _points.push(x);
    _points.push(y);
    _points.push(z);
    amount--;
  }
}

function points(asArray) {
  if (asArray) {
    let arr = [];
    for (let n = 0; n <= _points.length - 3; n += 3) {
      arr.push([_points[n], _points[n + 1], _points[n + 2]]);
    }

    return arr;
  } else {
    return _points;
  }
}

module.exports = { init, next, points };
