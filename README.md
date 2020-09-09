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
const points = lorenz.points(); // Return array of points
```

If you want to visualize the actual points in a third dimensional you can use [Three.JS](https://threejs.org/) or any other 3D engine.

You can also see a practical use of this module here: http://attractors.adelriosantiago.com/

[![attractor](example.gif "Lorenz")](http://attractors.adelriosantiago.com/)
