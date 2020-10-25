# lorenz-attractor-3d

This generates the Lorenz attractor points for a third dimensional space.

## Installation

`npm install --save lorenz-attractor-3d`

## Usage

Use `import lorenz = require("lorenz-attractor-3d")` or `<script src="lorenz.build.js"></script>` for browsers. Then,

```js
lorenz.init(); // Init with default values, or
//lorenz.init(sigma, beta, rho, initX, initY, initZ, stepSize) //Init with other initial conditions
lorenz.next(); // Generate next value, or
//lorenz.next(999) // Generate next 999 values
const points = lorenz.points(); // Return array of points [x1, y1, z1, x2, y2, z2, ...]
const pontsAsArray = lorenz.points(true); // Return points [[x1, y1, z1], [x2 ,y2, z2], ...]
const pontsAsVector3 = lorenz.points(true, THREE.Vector3); // Return points as another object [THREE.Vector3(x1, y1, z1), THREE.Vector3(x2 ,y2, z2), ...]. Useful when using libraries like THREE.js
```

Init settings:

- `sigma (default `10.0`): Lorenz sigma value
- `beta` (`2.667`): Lorenz beta value
- `rho` (`28.0`): Lorenz rho value
- `arrType` (`Float32Array`): Default point type
- `max` (`10000`): Maximum number of points that can be generated, _note that further `next` calls will silently do nothing_
- `x` (`1`): Starting X position
- `y` (`1`): Starting Y position
- `z` (`1`): Starting Z position
- `stepSize` (`0.004`): Multiplies each X, Y, Z by this value, avoids early bailouts
- `bailout` (`1e10`): Beyond this number, the algorithm stops processing more values

If you want to visualize the actual points in a third dimensional you can use [Three.JS](https://threejs.org/) or any other 3D engine.

Demo:
[![attractor](example.gif "Lorenz")](http://attractors.adelriosantiago.com/)
