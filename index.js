const Matrix = require("ml-matrix");

module.exports = {
  nmf: nmf
}


/**
 * Compute the NMF of a matrix V, i.e the matrix W and H => A ~= W.H
 * @param {Matrix} V - Matrix to factorize
 * @param {Object} options - options can include the parameters K (width of the Matrix W and height of the Matrix H), Winit (Init matrix of W), Hinit (Init matrix of H), tol (tolerance - default is 0.001) and maxIter (maximum of iterations before stopping - default is 100)
 * @return {Object} WH - object with the format {W: ..., H: ...}. W and H are the results (i.e A ~= W.H)
 */

function nmf(V,options){
  const n = V.rows;
  const m = V.columns;
  const {
    K: 2,
    Winit: Matrix.zeros(m, K),
    Hinit: Matrix.zeros(K, n),
    tol: 0.001,
    maxIter: 100
  } = options;

  var W = Winit;
  var H = Hinit;
  var gradW = W.mmul(H.mmul(H.transpose())) - V.mmul(H.transpose());
  var gradH = W.transpose().mmul(W).mmul(H) - W.transpose().mmul(V);
  
  var initgrad = concatMatrix(gradW, gradH.transpose(), 'H');
  
  initgrad = norm2(initgrad);
  var tolW = max(0.001, tol)*initGrad;
  var tolH = tolW;

  for(var i; i < maxIter; i++){
    var projnorm = norm2(concatMatrix(gradW[logical_or(gradW.min() < 0, W.min() > 0)], gradH[logical_or(gradH.min() < 0, H.min() > 0)]));
    if(projnrom < tol*initgrad){
      break;
    }
    tmp = nlssubprob(V.transpose(),H.transpose(),W.transpose(),tolW,1000)
    W = tmp.M;
    gradW = tmp.grad;
    iterW = tmp.iter;

    W = W.transpose();
    gradW = gradW.transpose();
  
    if (iterW==1){
      tolW = 0.1 * tolW
    }
  
    tmp = nlssubprob(V,W,H,tolH,1000);
    H = tmp.M;
    gradH = tmp.grad;
    iterH = tmp.iter;
    if(iterH==1){
      tolH = 0.1 * tolH;
    }
  }
  console.log('\nIter = '+iter);
  return {W: W,H: H}
}

function nlssubprob(V, W, Hinit, tol, maxIter){
  var H = Hinit
  var WtV = W.transpose().mmul(V);
  var WtW = W.transpose().mmul(W);
  var grad = 0;
  var alpha = 1;
  var beta = 0.1;

  for(var i = 0; i < maxIter; i++){
    grad = WtW.mmul(H) - WtV;
    var projgrad = norm2(grad[logical_or(grad < 0, H >0)]);
    if(projgrad < tol){
      break;
    }
    for inner_iter in xrange(1,20){
      var Hn = H - alpha*grad
      Hn = replaceElement(Hn, Hn > 0, 0)
      var d = Hn-H;
      var gradd = sum(grad.mmul(d)):
      var dQd = sum(dot(WtW,d) * d)
      var suff_decr = 0.99*gradd + 0.5*dQd < 0;
      if(inner_iter == 1){
        var decr_alpha = suff_decr == 1 ? 0 : 1; 
        var Hp = H;
      }
      if(decr_alpha == 1){
        if(suff_decr == 1){
          H = Hn; 
          break;
        }
        else{
          alpha = alpha * beta;
        }
      }
      else{
        if((suff_decr != 1) || (Hp == Hn).all()){
          H = Hp; 
          break;
        }
        else{
          alpha = alpha/beta; 
          Hp = Hn;
        }
      }
    }
  }

  return {M: H, grad: grad, iter: i};
}


function norm2(A){
  var result = 0;
  for(var i = 0; i < A.rows; i++){
    for(var j = 0; j < A.columns; j++){
      result = result + Math.abs(A.get(i,j))**2;
    }
  }
  return Math.sqrt(result);
}

function concatMatrix(A, B, direction='H'){
  var result = A;
  if(direction == 'H'){
    for(var i = 0; i < B.columns; i++){
      result = result.addColumnVector(B.getColumn(i));
    }
  }
  else{
    for(var i = 0; i < B.rows; i++){
      result = result.addRowVector(B.getRow(i));
    }
  }
  return result;
}


function logical_or(c1, c2){
  return (c1 || c2) ? 1 : 0;
}

