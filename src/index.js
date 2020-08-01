const { Matrix } = require('ml-matrix');

module.exports = {
  nmf: nmf,
};

/**
 * Compute the NMF of a matrix V, i.e the matrix W and H => A ~= W.H
 * @param {Matrix} V - Matrix to factorize
 * @param {Object} options - options can include the parameters K (width of the Matrix W and height of the Matrix H), Winit (Init matrix of W), Hinit (Init matrix of H), tol (tolerance - default is 0.001) and maxIter (maximum of iterations before stopping - default is 100)
 * @return {Object} WH - object with the format {W: ..., H: ...}. W and H are the results (i.e A ~= W.H)
 */

function nmf(V, options) {
  const n = V.rows;
  const m = V.columns;
  const {
    K = 2,
    Winit = Matrix.random(n, K),
    Hinit = Matrix.random(K, m),
    tol = 0.001,
    maxIter = 100,
  } = options;

  let W = Winit;
  let H = Hinit;
  let WH = W.mmul(H);
  let gradW = Matrix.sub(WH, V).mmul(H.transpose());
  let gradH = W.transpose().mmul(Matrix.sub(WH, V));

  let initgrad = norm2(gradW.to1DArray().concat(gradH.transpose().to1DArray()));
  let tolW = Math.max(0.001, tol) * initgrad;
  let tolH = tolW;
  let i;
  for (i = 1; i < maxIter; i++) {
    let projnorm = norm2(
      selectElementsFromMatrix(
        gradW,
        logicalOrMatrix(
          elementsMatrixInferiorZero(gradW),
          elementsMatrixSuperiorZero(W),
        ),
      ).concat(
        selectElementsFromMatrix(
          gradH,
          logicalOrMatrix(
            elementsMatrixInferiorZero(gradH),
            elementsMatrixSuperiorZero(H),
          ),
        ),
      ),
    );
    // console.log('projnorm', projnorm);
    if (projnorm < tol * initgrad) {
      console.log('it break');
      break;
    }
    let tmp = nlssubprob(
      V.transpose(),
      H.transpose(),
      W.transpose(),
      tolW,
      100,
    );
    W = tmp.M;
    // console.log(tmp.iter, W);
    gradW = tmp.grad;
    let iterW = tmp.iter;

    W = W.transpose();
    gradW = gradW.transpose();

    if (iterW === 1) {
      tolW = 0.1 * tolW;
    }

    tmp = nlssubprob(V, W, H, tolH, 100);
    H = tmp.M;
    gradH = tmp.grad;
    let iterH = tmp.iter;
    if (iterH === 1) {
      tolH = 0.1 * tolH;
    }
  }
  return { W: W, H: H };
}

/**
 * @private
 * Solve the subproblem by the projected gradient algorithm
 * @param {Matrix} V
 * @param {Matrix} W
 * @param {Matrix} Hinit
 * @param {number} tol (between 0 and 1)
 * @param {number} maxIter
 * @return {object} return value has the format {W: , H: }
 */

function nlssubprob(V, W, Hinit, tol, maxIter) {
  let H = Hinit;
  let WtV = W.transpose().mmul(V);
  let WtW = W.transpose().mmul(W);
  let grad;
  let alpha = 1;
  let beta = 0.01;
  let decrAlpha;
  let Hp;
  let numberIterations;
  for (let iter = 1; iter < maxIter; iter++) {
    numberIterations = iter;
    grad = Matrix.sub(WtW.mmul(H), WtV);
    let projgrad = norm2(
      selectElementsFromMatrix(
        grad,
        logicalOrMatrix(
          elementsMatrixInferiorZero(grad),
          elementsMatrixSuperiorZero(H),
        ),
      ),
    );
    if (projgrad < tol) {
      break;
    }
    for (let innerIter = 1; innerIter < 20; innerIter++) {
      let Hn = Matrix.sub(H, Matrix.mul(grad, alpha));
      Hn = replaceElementsMatrix(Hn, elementsMatrixInferiorZero(Hn), 0);
      //compute condition 13 to search optimium alpha
      let d = Matrix.sub(Hn, H);
      let gradd = sumElements(multiplyElementByElement(d, grad));
      let dQd = sumElements(multiplyElementByElement(WtW.mmul(d), d));
      let suffDecr = 0.99 * gradd + 0.5 * dQd <= 0; //condition 13 with sigma = 0.01

      if (innerIter === 1) {
        decrAlpha = !suffDecr;
        Hp = H.clone();
      }
      if (decrAlpha) {
        if (suffDecr) {
          H = Hn.clone();
          break;
        } else {
          alpha *= beta;
        }
      } else {
        if (!suffDecr || matrixEqual(Hp, Hn)) {
          H = Hp.clone();
          break;
        } else {
          alpha /= beta;
          Hp = Hn.clone();
        }
      }
    }
  }
  return { M: H, grad: grad, iter: numberIterations };
}

/**
 * @private
 * Return the frobenius norm of an array
 * @param {Array<number>} A
 * @return {number} the frobenius norm
 */

function norm2(A) {
  let result = 0;
  for (let i = 0; i < A.length; i++) {
    result = result + Math.abs(A[i]) ** 2;
  }
  return Math.sqrt(result);
}

/**
 * @private
 * Return a 1D-Array with the elements of the matrix X which are superior than 0
 * @param {Matrix} X
 * @return {Array<number>} elements superior than 0
 */

function elementsMatrixSuperiorZero(X) {
  let newArray = new Array(X.rows);
  for (let i = 0; i < newArray.length; i++) {
    newArray[i] = new Array(X.columns);
    for (let j = 0; j < X.columns; j++) {
      newArray[i][j] = X.get(i, j) > 0;
    }
  }
  return newArray;
}

/**
 * @private
 * Return a 1D-Array with the elements of the matrix X which are inferior than 0
 * @param {Matrix} X
 * @return {Array<number>} elements inferior than 0
 */

function elementsMatrixInferiorZero(X) {
  let newArray = new Array(X.rows);
  for (let i = 0; i < newArray.length; i++) {
    newArray[i] = new Array(X.columns);
    for (let j = 0; j < X.columns; j++) {
      newArray[i][j] = X.get(i, j) < 0;
    }
  }
  return newArray;
}

/**
 * @private
 * Take a matrix and a 2D-array of booleans (same dimensions than the matrix) and return the elements of the matrix which corresponds to a value true in the 2D-array of booleans.
 * @param {Matrix} X
 * @param {Array<Array<boolean>>} arrayBooleans
 * @return {Array<number>} elements selected
 */

function selectElementsFromMatrix(X, arrayBooleans) {
  if (
    X.rows !== arrayBooleans.length ||
    X.columns !== arrayBooleans[0].length
  ) {
    throw new Error('Error of dimension');
  }
  let newArray = [];
  let rows = X.rows;
  let columns = X.columns;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < columns; c++) {
      if (arrayBooleans[r][c]) {
        newArray.push(X.get(r, c));
      }
    }
  }
  return newArray;
}

/**
 * @private
 * Take a matrix, a 2D-array of booleans (same dimensions than the matrix) and a value. Replace the elements of the matrix which corresponds to a value true in the 2D-array of booleans by the value given into parameter. Return a matrix.
 * @param {Matrix} X
 * @param {Array<Array<boolean>>} arrayBooleans
 * @param {number} value
 * @return {Matrix} Matrix which the replaced elements.
 */

function replaceElementsMatrix(X, arrayBooleans, value) {
  if (
    X.rows !== arrayBooleans.length ||
    X.columns !== arrayBooleans[0].length
  ) {
    throw new Error('Error of dimension');
  }
  let rows = X.rows;
  let columns = X.columns;
  let newMatrix = new Matrix(X);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < columns; c++) {
      if (arrayBooleans[r][c]) {
        newMatrix.set(r, c, value);
      }
    }
  }
  return newMatrix;
}

/**
 * @private
 * Take 2 2D-array of booleans (same dimensions) and return another 2D-Array with the result or the OR operator (each element of the result is the result of the OR operation of the corresponding elements in the 2 arrays given into parameters).
 * @param {Array<Array<boolean>>} m1
 * @param {Array<Array<boolean>>} m2
 * @return {Array<Array<boolean>>} 2D-Array with the result of the OR operations.
 */

function logicalOrMatrix(m1, m2) {
  if (m1.length !== m2.length || m1[0].length !== m2[0].length) {
    throw new Error('Error of dimension');
  }
  let newArray = new Array(m1.length);
  for (let i = 0; i < newArray.length; i++) {
    newArray[i] = new Array(m1[0].length);
    for (let j = 0; j < m1[0].length; j++) {
      newArray[i][j] = m1[i][j] || m2[i][j];
    }
  }
  return newArray;
}

/**
 * @private
 * Return the sum of every elements of the matrix
 * @param {Matrix} X
 * @return {number}
 */

function sumElements(X) {
  let rows = X.rows;
  let columns = X.columns;
  let result = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < columns; c++) {
      result += X.get(r, c);
    }
  }
  return result;
}

/**
 * @private
 * Matrix multiplication element-wise.
 * @param {Matrix} m1
 * @param {Matrix} m2
 * @return {Matrix}
 */

function multiplyElementByElement(m1, m2) {
  if (m1.rows !== m2.rows || m1.columns !== m2.columns) {
    throw new Error('Error of dimension');
  }
  let rows = m1.rows;
  let columns = m1.columns;
  let newMatrix = Matrix.zeros(rows, columns);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < columns; c++) {
      newMatrix.set(r, c, m1.get(r, c) * m2.get(r, c));
    }
  }
  return newMatrix;
}

/**
 * @private
 * Return if two matrix are equals (i.e each element is equal to the corresponding element of the other matrix).
 * @param {Matrix} m1
 * @param {Matrix} m2
 * @return {boolean}
 */

function matrixEqual(m1, m2) {
  if (m1.rows !== m2.rows || m1.columns !== m2.columns) {
    throw new Error('Error of dimension');
  }
  let rows = m1.rows;
  let columns = m1.columns;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < columns; c++) {
      if (m1.get(r, c) !== m2.get(r, c)) {
        return false;
      }
    }
  }
  return true;
}
