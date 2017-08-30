Non-negative Matrix Factorization (NMF)
=======================================

Implementation of the projected gradient methods for NMF. [Wikipedia](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)

Usage
=====

```js
const NMF = require('../src');
const {Matrix} = require("ml-matrix");

let w = new Matrix([[1,2,3],[4,5,6]]);
let h = new Matrix([[1,2],[3,4],[5,6]]);
let winit = new Matrix([[1,1,3],[4,5,6]]);
let hinit = new Matrix([[1,1],[3,4],[5,6]]);

let v = w.mmul(h);

const options = {
  Winit: winit, 
  Hinit: hinit,
  tol: 0.001, 
  maxIter: 10
}

let result = NMF.nmf(v, options);
let w0 = result.W;
let h0 = result.H;
console.log('W computed :', w0);
console.log('H computed :', h0);
console.log('W*H :', w0.mmul(h0));
console.log('expected :', v);
```

References
==========

C.-J. Lin. Projected gradient methods for non-negative matrix factorization. Neural Computation, 19(2007), 2756-2779.
