// const { Matrix, SingularValueDecomposition } = require('ml-matrix');
import { Matrix, SingularValueDecomposition } from 'ml-matrix';

// module.exports = initialisation;

/**
 * Initialize the parameter for NMF
 * @param {Matrix} V
 * @param {number} k
 * @param {boolean} svdInitialisation
 */

export function initialisation(V, options = {}) {
  let {
    k,
    seed = 2222,
    maxV = V.max(),
    wInit,
    hInit,
    svdInitialisation = false,
  } = options;

  const n = V.rows;
  const m = V.columns;

  if (wInit && hInit) {
    Matrix.checkMatrix(wInit);
    Matrix.checkMatrix(hInit);
    if (wInit.columns !== k || wInit.rows !== n) {
      throw new Error(`guess W matrix does not match rows ${n} columns ${k}`);
    }
    if (hInit.columns !== m || hInit.rows !== k) {
      throw new Error(`guess H matrix does not match rows ${k} columns ${m}`);
    }
    return { W: wInit, H: hInit };
  } else if (svdInitialisation === true) {
    return initSVD(V, { k });
  } else {
    let random = () => {
      return 1e-8 + Math.random(seed) * maxV;
    };
    let W = Matrix.rand(V.rows, k, { random });
    let H = Matrix.rand(k, V.columns, { random });
    return { W, H };
  }
}

/**
 * Choose appropriate rank for nmf and good starting matrix (corresponding to given rank)
 * @param {Matrix} V
 * @param {object} options
 * @param {number} [options.k]
 */

function initSVD(V, options = {}) {
  const { k = 0 } = options;
  let svdV = new SingularValueDecomposition(V, { autoTranspose: false });
  let rank;
  if (k === 0) {
    let sumSingular = arrSum(svdV.s);
    let i = 0;
    let normalizedSum = 0;
    while (i < svdV.s.length && normalizedSum < 0.9) {
      normalizedSum += svdV.s[i] / sumSingular;
      i++;
    }
    rank = i;
  } else {
    rank = k;
  }
  let s = new Array(rank);
  for (let j = 0; j < rank; j++) {
    s[j] = Math.sqrt(svdV.s[j]);
  }

  let sqrtRightS = Matrix.diag(s, rank, V.columns);
  let sqrtLeftS = Matrix.diag(s, V.rows, rank);

  let W0 = svdV.U.subMatrix(0, V.rows - 1, 0, rank - 1).mmul(sqrtLeftS);
  let H0 = sqrtRightS.mmul(svdV.V.transpose());

  let W = W0.abs();
  let H = H0.abs();

  return { W, H, rank };
}

function arrSum(arr) {
  return arr.reduce(function (a, b) {
    return a + b;
  }, 0);
}
