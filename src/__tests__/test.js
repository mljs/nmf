import { Matrix } from 'ml-matrix';
import { toBeDeepCloseTo } from 'jest-matcher-deep-close-to';
import { nmf as NMF } from '../index';

expect.extend({ toBeDeepCloseTo });

describe('NMF test', () => {
  it('use case 1', async () => {
    let w = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    let h = new Matrix([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    let winit = new Matrix([
      [1, 1, 3],
      [4, 5, 6],
    ]);
    let hinit = new Matrix([
      [1, 1],
      [3, 4],
      [5, 6],
    ]);

    let v = w.mmul(h);

    const options = {
      wInit: winit,
      hInit: hinit,
      tol: 0.001,
      maxIter: 10,
    };

    const result = NMF(v, options);
    const w0 = result.W;
    const h0 = result.H;
    expect(w0.mmul(h0).to2DArray()).toBeDeepCloseTo(v.to2DArray(), 1);
  });
});
