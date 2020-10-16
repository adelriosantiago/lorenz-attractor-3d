# lorenz-attractor-3d

This generates the Lorenz attractor points for a third dimensional space.

## Installation

`npm install lorenz-attractor-3d --save`

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

If you want to visualize the actual points in a third dimensional you can use [Three.JS](https://threejs.org/) or any other 3D engine.

You can also see a practical use of this module here: http://attractors.adelriosantiago.com/

[![attractor](example.gif "Lorenz")](http://attractors.adelriosantiago.com/)
