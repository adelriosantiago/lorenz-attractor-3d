(function(f){if(typeof exports==="object"&&typeof module!=="undefined"){module.exports=f()}else if(typeof define==="function"&&define.amd){define([],f)}else{var g;if(typeof window!=="undefined"){g=window}else if(typeof global!=="undefined"){g=global}else if(typeof self!=="undefined"){g=self}else{g=this}g.lorenz = f()}})(function(){var define,module,exports;return (function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
let _points = [];
let _nx, _ny, _nz;
let _cursor;

let x, y, z;
let sigma, beta, rho;
let stepSize;

function init(
  _sigma = 10.0,
  _beta = 2.667,
  _rho = 28.0,
  arrType = Float32Array,
  max = 10000,
  _x = 1,
  _y = 1,
  _z = 1,
  _stepSize = 0.004,
  _bailout = 1e10
) {
  _points = new arrType(max * 3);
  sigma = _sigma;
  beta = _beta;
  rho = _rho;
  x = _x;
  y = _y;
  z = _z;
  stepSize = _stepSize;
  bailout = _bailout;
  _cursor = 0;
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

    _points[_cursor] = x;
    _points[_cursor + 1] = y;
    _points[_cursor + 2] = z;
    amount--;
    _cursor += 3;
  }
}

function points(asArray, type = Array) {
  if (asArray) {
    let arr = [];
    for (let n = 0; n <= _points.length - 3; n += 3) {
      arr.push(new type(_points[n], _points[n + 1], _points[n + 2]));
    }

    return arr;
  } else {
    return _points;
  }
}

function cursor() {
  return _cursor / 3;
}

module.exports = { init, next, points, cursor };

},{}]},{},[1])(1)
});
