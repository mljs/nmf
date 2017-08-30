'use strict';

const {Matrix} = require('ml-matrix');

module.exports = {
    nmf: nmf
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
        Winit = Matrix.zeros(m, K),
        Hinit = Matrix.zeros(K, n),
        tol = 0.001,
        maxIter = 100
    } = options;

    let W = Winit;
    let H = Hinit;
    let gradW = Matrix.sub(W.mmul(H.mmul(H.transpose())), V.mmul(H.transpose()));
    let gradH = Matrix.sub(W.transpose().mmul(W).mmul(H), W.transpose().mmul(V));

    let initgrad = norm2(gradW.to1DArray().concat(gradH.transpose().to1DArray()));
    let tolW = Math.max(0.001, tol) * initgrad;
    let tolH = tolW;

    for (let i = 1; i < maxIter; i++) {
        let projnorm = norm2(selectElementsFromMatrix(gradW, logicalOrMatrix(elementsMatrixInferiorZero(gradW), elementsMatrixSuperiorZero(W))).concat(selectElementsFromMatrix(gradH, logicalOrMatrix(elementsMatrixInferiorZero(gradH), elementsMatrixSuperiorZero(H)))));
        if (projnorm < tol * initgrad) {
            break;
        }
        let tmp = nlssubprob(V.transpose(), H.transpose(), W.transpose(), tolW, 1000);
        W = tmp.M;
        gradW = tmp.grad;
        let iterW = tmp.iter;

        W = W.transpose();
        gradW = gradW.transpose();

        if (iterW === 1) {
            tolW = 0.1 * tolW;
        }

        tmp = nlssubprob(V, W, H, tolH, 1000);
        H = tmp.M;
        gradH = tmp.grad;
        let iterH = tmp.iter;
        if (iterH === 1) {
            tolH = 0.1 * tolH;
        }
    }
    //console.log('\nIter = '+i);
    return {W: W, H: H};
}

function nlssubprob(V, W, Hinit, tol, maxIter) {
    let H = Hinit;
    let WtV = W.transpose().mmul(V);
    let WtW = W.transpose().mmul(W);
    let grad;
    let alpha = 1;
    let beta = 0.1;
    let decrAlpha;
    let Hp;
    let numberIterations;
    maxIter = 2;
    for (let iter = 1; iter < maxIter; iter++) {
        numberIterations = iter;
        grad = Matrix.sub(WtW.mmul(H), WtV);
        //let projgrad = norm2(selectElementsFromMatrix(grad, logicalOrMatrix(elementsMatrixInferiorZero(grad), elementsMatrixSuperiorZero(H))));
        for (let innerIter = 1; innerIter < 20; innerIter++) {
            let Hn = Matrix.sub(H, Matrix.mul(grad, alpha));
            Hn = replaceElementsMatrix(Hn, elementsMatrixSuperiorZero(Hn), 0);
            let d = Matrix.sub(Hn, H);
            let gradd = sumElements(multiplyElementByElement(d, grad));
            let dQd = sumElements(multiplyElementByElement(WtW.mmul(d), d));
            let suffDecr = 0.99 * gradd + 0.5 * dQd < 0;
            if (innerIter === 1) {
                decrAlpha = !suffDecr;
                Hp = H.clone();
            }
            if (decrAlpha) {
                if (suffDecr) {
                    H = Hn.clone();
                    break;
                } else {
                    alpha = alpha * beta;
                }
            } else {
                if (!suffDecr || matrixEqual(H, Hp)) {
                    H = Hp.clone();
                    break;
                } else {
                    alpha = alpha / beta;
                    Hp = H.clone();
                }
            }
        }

        /*if (iter === maxIter) {
            console.log('Max iterations in nlssubprob');
        }*/
    }
    return {M: H, grad: grad, iter: numberIterations};
}


function norm2(A) {
    let result = 0;
    for (let i = 0; i < A.length; i++) {
        result = result + Math.abs(A[i]) ** 2;
    }
    return Math.sqrt(result);
}

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

function selectElementsFromMatrix(X, arrayBooleans) {
    if (X.rows !== arrayBooleans.length || X.columns !== arrayBooleans[0].length) {
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

function replaceElementsMatrix(X, arrayBooleans, value) {
    if (X.rows !== arrayBooleans.length || X.columns !== arrayBooleans[0].length) {
        throw new Error('Error of dimension');
    }
    let rows = X.rows;
    let columns = X.columns;
    let newMatrix = new Matrix(X);
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < columns; c++) {
            if (!arrayBooleans[r][c]) {
                newMatrix.set(r, c, value);
            }
        }
    }
    return newMatrix;
}

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

function sumElements(X) {
    let rows = X.rows;
    let columns = X.columns;
    let result = 0;
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < columns; c++) {
            result = result + X.get(r, c);
        }
    }
    return result;
}

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
