#![feature(zero_one)]

#![no_std]

extern crate peano;

use core::marker::PhantomData;
use core::mem;
use core::num::*;
use core::ops::*;
use core::ptr;
use core::slice;

use peano::Natural;

pub trait Signature {
    type TotalDim: Natural;
    type EllipticDim: Natural;
    type HyperbolicDim: Natural;
}

mod basis {
    use peano::Natural;

    use super::Signature;

    /// Multiplies 2 basis elements.
    /// The indices of the basis elements have a bit for each element of the
    /// generating set, which are in such an order, from LSB to MSB:
    /// hyperbolic, elliptic, flat
    /// Returns None if product is zero.
    pub fn mul<S: Signature>(i: usize, j: usize) -> Option<(bool /* answer is negative */, usize)> {
        let nulMask = !0<<(S::EllipticDim::toUsize()+S::HyperbolicDim::toUsize());
        let negMask = !0<<S::HyperbolicDim::toUsize();
        if i&j&nulMask != 0 { None } else { Some((((i&j&negMask).count_ones()&1 != 0) ^ reversalParity(j) ^ parityToBringTogetherLikeFactors(i, j), i^j)) }
    }

    fn reversalParity(k: usize) -> bool { k.count_ones()&2 != 0 }

    // operands start with LSBs together
    fn parityToBringTogetherLikeFactors(mut i: usize, mut j: usize) -> bool {
        let mut p = 0;
        while i != 0 && j != 0 {
            let n = (i&j).trailing_zeros();
            let f = |k: usize| k.count_ones() & (k>>n).count_ones();
            p ^= f(i)^f(j);
            i = (i>>n)^1;
            j = (j>>n)^1;
        }
        p&1 != 0
    }
}

pub trait LogArrayLength<A>: Natural { type Array; }

impl<A> LogArrayLength<A> for peano::Zero { type Array = A; }
impl<A, N: LogArrayLength<A>> LogArrayLength<A> for peano::Succ<N> { type Array = (N::Array, N::Array); }

struct Array<A, N: LogArrayLength<A>>(N::Array);

impl<A: Zero, N: LogArrayLength<A>> Zero for Array<A, N> {
    fn zero() -> Self {
        unsafe {
            let mut a: Self = mem::uninitialized();
            for k in 0..1<<N::toUsize() {
                ptr::write((&mut a as *mut Self as *mut A).offset(k), Zero::zero());
            }
            a
        }
    }
}

impl<A, N: LogArrayLength<A>> Deref for Array<A, N> {
    type Target = [A];
    fn deref(&self) -> &[A] {
        unsafe { slice::from_raw_parts(self as *const Self as *const A, N::toUsize()) }
    }
}

impl<A, N: LogArrayLength<A>> DerefMut for Array<A, N> {
    fn deref_mut(&mut self) -> &mut [A] {
        unsafe { slice::from_raw_parts_mut(self as *mut Self as *mut A, N::toUsize()) }
    }
}

impl<A, N: LogArrayLength<A>, I> Index<I> for Array<A, N> where [A]: Index<I> {
    type Output = <[A] as Index<I>>::Output;
    fn index(&self, k: I) -> &Self::Output { &(self as &[A])[k] }
}

impl<A, N: LogArrayLength<A>, I> IndexMut<I> for Array<A, N> where [A]: IndexMut<I> {
    fn index_mut(&mut self, k: I) -> &mut Self::Output { &mut (self as &mut [A])[k] }
}

pub struct Complex<A, S = (peano::Zero, peano::Succ<peano::Zero>, peano::Zero)>(PhantomData<S>, Array<A, S::TotalDim>) where S: Signature, S::TotalDim: LogArrayLength<A>;

impl<A: Copy + Neg<Output = A>, S: Signature> Complex<A, S> where S::TotalDim: LogArrayLength<A> {
    #[inline] pub fn conjugate(self) -> Self {
        let Complex(_, mut a) = self;
        for p in a.iter_mut() { *p = (*p).neg() }
        Complex(PhantomData, a)
    }
}

impl<A: Copy + Add<Output = A>, S: Signature> Add for Complex<A, S> where S::TotalDim: LogArrayLength<A> {
    type Output = Self;
    #[inline] fn add(self, Complex(_, b): Self) -> Self {
        let Complex(_, mut a) = self;
        for i in 0..S::TotalDim::toUsize() { a[i] = a[i] + b[i] }
        Complex(PhantomData, a)
    }
}

impl<A: Zero, S: Signature> Zero for Complex<A, S> where S::TotalDim: LogArrayLength<A> {
    #[inline] fn zero() -> Self { Complex(PhantomData, Zero::zero()) }
}

impl<A: Copy + Add<Output = A> + Sub<Output = A> + Mul<Output = A>, S: Signature> Mul for Complex<A, S> where S::TotalDim: LogArrayLength<A> {
    type Output = Self;
    #[inline] fn mul(self, Complex(_, b): Self) -> Self {
        let Complex(_, mut a) = self;
        for p in a.iter_mut() { *p = *p*b[0] }
        for j in 1..S::TotalDim::toUsize() {
            for i in 0..S::TotalDim::toUsize() {
                if let Some((p, k)) = basis::mul::<S>(i, j) {
                    a[k] = (if p { Sub::sub as fn(A, A) -> A } else { Add::add as fn (A, A) -> A })(a[k], a[i]*b[j])
                }
            }
        }
        Complex(PhantomData, a)
    }
}

impl<A: Zero + One, S: Signature> One for Complex<A, S> where S::TotalDim: LogArrayLength<A> {
    #[inline] fn one() -> Self {
        let mut a: Array<A, S::TotalDim> = Zero::zero();
        a[0] = One::one();
        Complex(PhantomData, a)
    }
}
