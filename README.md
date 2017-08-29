Non-negative Matrix Factorization (NMF)
=======================================

Implementation of the projected gradient methods for NMF.

Usage
=====

```js
const NMF = require('../src');
const {Matrix} = require("ml-matrix");

let w = new Matrix([[1,2,3],[4,5,6]]);
let h = new Matrix([[1,2],[3,4],[5,6]]);
let w2 = new Matrix([[1,1,3],[4,5,6]]);
let h2 = new Matrix([[1,1],[3,4],[5,6]]);

let v = w.mmul(h);

const options = {
  Winit: w2, 
  Hinit: h2,
  tol: 0.001, 
  maxIter: 10
}

tmp = NMF.nmf(v, options);
w0 = tmp.W;
h0 = tmp.H;
console.log('W computed :', w0);
console.log('H computed :', h0);
console.log('W*H :', w0.mmul(h0));
console.log('expected :', v);
```

References
==========

C.-J. Lin. Projected gradient methods for non-negative matrix factorization. Neural Computation, 19(2007), 2756-2779.
