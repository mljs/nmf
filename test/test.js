const NMF = require('../src');
const Matrix = require("ml-matrix");

var w = Matrix([[1,2,3],[4,5,6]]);
var h = Matrxi([[1,2],[3,4],[5,6]]);
var w2 = Matrix([[1,1,3],[4,5,6]]);
var h2 = Matrix([[1,1],[3,4],[5,6]]);

var v = w.mmul(h);

const options = {
  Winit: w2, 
  Hinit: h2,
  tol: 0.001, 
  maxIter: 10
}

tmp = NMF.nmf(v, options);
w0 = tmp.W;
h0 = tmp.H;
console.log(w0);
console.log(h1);
