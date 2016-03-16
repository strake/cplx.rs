#![feature(const_fn)]
#![feature(zero_one)]

#![no_std]

extern crate typenum;

use core::cmp::*;
use core::marker::PhantomData;
use core::num::*;
use core::ops::*;
use typenum::consts::{ P1, N1 };
use typenum::int::{ Integer, Z0 };

pub trait Sign<A> : Integer { fn sign(A) -> A; }
impl<A: Zero>            Sign<A> for Z0 { fn sign(_: A) -> A { Zero::zero() } }
impl<A>                  Sign<A> for P1 { fn sign(a: A) -> A { a } }
impl<A: Neg<Output = A>> Sign<A> for N1 { fn sign(a: A) -> A { a.neg() } }

/// Cayley-Dickson construction
pub struct Complex<A, S: Sign<A> = N1>(PhantomData<S>, A, A);

impl<S: Sign<A>, A> Complex<A, S> {
    #[inline] pub const fn from_rect(re: A, im: A) -> Self { Complex(PhantomData, re, im) }
    #[inline] pub fn to_rect(self) -> (A, A) { let Complex(_, re, im) = self; (re, im) }
}

#[inline] pub const fn from_rect<S: Sign<A>, A>(re: A, im: A) -> Complex<A, S> { Complex::<A, S>::from_rect(re, im) }

impl<S: Sign<A>, A: Clone> Clone for Complex<A, S> {
    #[inline] fn clone(&self) -> Self {
        let &Complex(_, ref a, ref b) = self;
        Complex(PhantomData, a.clone(), b.clone())
    }
}

impl<S: Sign<A>, A: Copy> Copy for Complex<A, S> {}

impl<S: Sign<A>, A: PartialEq> PartialEq for Complex<A, S> {
    #[inline] fn eq(&self, &Complex(_, ref c, ref d): &Self) -> bool {
        let &Complex(_, ref a, ref b) = self;
        (a, b) == (c, d)
    }
}

impl<S: Sign<A>, A: Eq> Eq for Complex<A, S> {}

pub trait Conjugable {
    fn conjugate(self) -> Self;
}

impl<S: Sign<A>, A: Add<Output = A> + Neg<Output = A> + Conjugable> Conjugable for Complex<A, S> {
    #[inline] fn conjugate(self) -> Self {
        let Complex(_, a, b) = self;
        Complex(PhantomData, a.conjugate(), b.neg())
    }
}

macro_rules! impl_Conjugable_id { ($t: ty) => (impl Conjugable for $t { fn conjugate(self) -> Self { self } }) }
impl_Conjugable_id!(());
impl_Conjugable_id!(isize);
impl_Conjugable_id!(i8);
impl_Conjugable_id!(i16);
impl_Conjugable_id!(i32);
impl_Conjugable_id!(i64);
impl_Conjugable_id!(f32);
impl_Conjugable_id!(f64);

impl<S: Sign<A>, A: Zero + Add<Output = A>> Zero for Complex<A, S> {
    #[inline] fn zero() -> Self { Complex(PhantomData, Zero::zero(), Zero::zero()) }
}

impl<S: Sign<A>, A: Zero + Add<Output = A>> Add for Complex<A, S> {
    type Output = Self;
    #[inline] fn add(self, Complex(_, c, d): Self) -> Self {
        let Complex(_, a, b) = self;
        Complex(PhantomData, a+c, b+d)
    }
}

impl<S: Sign<A>, A: Zero + Sub<Output = A>> Sub for Complex<A, S> {
    type Output = Self;
    #[inline] fn sub(self, Complex(_, c, d): Self) -> Self {
        let Complex(_, a, b) = self;
        Complex(PhantomData, a-c, b-d)
    }
}

impl<S: Sign<A>, A: Zero + Neg<Output = A>> Neg for Complex<A, S> {
    type Output = Self;
    #[inline] fn neg(self) -> Self {
        let Complex(_, a, b) = self;
        Complex(PhantomData, -a, -b)
    }
}

impl<S: Sign<A>, A: Zero + Add<Output = A> + One> One for Complex<A, S> {
    #[inline] fn one() -> Self { Complex(PhantomData, One::one(), Zero::zero()) }
}

impl<S: Sign<A>, A: Copy + Zero + Add<Output = A> + Conjugable + Mul<Output = A>> Mul for Complex<A, S> {
    type Output = Self;
    #[inline] fn mul(self, Complex(_, c, d): Self) -> Self {
        let Complex(_, a, b) = self;
        Complex(PhantomData, a*c+S::sign(d.conjugate()*b), d*a+b*c.conjugate())
    }
}

impl<S: Sign<A>, A: Copy + Zero + Add<Output = A> + Neg<Output = A> + Conjugable + Mul<Output = A> + Div<Output = A>> Div for Complex<A, S> {
    type Output = Self;
    #[inline] fn div(self, other: Self) -> Self {
        let Complex(_, a, b) =  self*other.conjugate();
        let Complex(_, c, _) = other*other.conjugate();
        Complex(PhantomData, a/c, b/c)
    }
}

#[cfg(test)] mod tests {
    use core::num::*;
    use typenum::consts::P1;
    use typenum::int::Z0;

    use super::*;

    #[test] fn complex_basis() {
        type T = Complex<isize>;
        let i: T = from_rect(0, 1);
        assert!(i*i == -T::one());
    }

    #[test] fn split_complex_basis() {
        type T = Complex<isize, P1>;
        let i: T = from_rect(0, 1);
        assert!(i*i == T::one());
    }

    #[test] fn dual_basis() {
        type T = Complex<isize, Z0>;
        let i: T = from_rect(0, 1);
        assert!(i*i == T::zero());
    }

    #[test] fn quaternion_basis() {
        type T = Complex<Complex<isize>>;
        let i: T = from_rect(from_rect(0, 1), from_rect(0, 0));
        let j: T = from_rect(from_rect(0, 0), from_rect(1, 0));
        let k: T = from_rect(from_rect(0, 0), from_rect(0, 1));
        assert!(i*j ==  k && j*k ==  i && k*i ==  j &&
                k*j == -i && j*i == -k && i*k == -j &&
                i*i == -T::one() && j*j == -T::one() && k*k == -T::one());
    }
}
