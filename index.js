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
  _stepSize = 0.004
) {
  _points = [];
  sigma = _sigma;
  beta = _beta;
  rho = _rho;
  x = _x;
  y = _y;
  z = _z;
  stepSize = _stepSize;
}

function next(amount = 1) {
  while (amount > 0) {
    _nx = sigma * (y - x);
    _ny = x * (rho - z) - y;
    _nz = x * y - beta * z;

    x += stepSize * _nx;
    y += stepSize * _ny;
    z += stepSize * _nz;

    _points.push([_nx, _ny, _nz]);
    amount--;
  }
}

function points() {
  return _points;
}

module.exports = { init, next, points };
