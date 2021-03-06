#![no_std]

#![feature(const_fn)]
#![feature(const_fn_union)]
#![feature(const_let)]
#![feature(untagged_unions)]

extern crate idem;
extern crate typenum;

use core::cmp::*;
use core::marker::PhantomData;
use core::ops::*;
use idem::*;
use typenum::consts::{ P1, N1 };
use typenum::int::{ Integer, Z0 };

pub trait Sign<A> : Integer { fn sign(A) -> A; }
impl<A>                  Sign<A> for P1 { fn sign(a: A) -> A { a } }
impl<A: Neg<Output = A>> Sign<A> for N1 { fn sign(a: A) -> A { a.neg() } }
impl<A: Zero>            Sign<A> for Z0 { fn sign(_: A) -> A { A::zero } }

/// Cayley-Dickson construction
#[derive(Debug)]
pub struct Complex<A, S: Sign<A> = N1>(PhantomData<S>, A, A);

impl<S: Sign<A>, A> Complex<A, S> {
    #[inline] pub const fn from_rect(re: A, im: A) -> Self { Complex(PhantomData, re, im) }
    #[inline] pub const fn into_rect(self) -> (A, A) {
        #[allow(unions_with_drop_fields)]
        union U<A, S: Sign<A>> { c: Complex<A, S> }
        let u = U { c: self };
        unsafe { (u.c.1, u.c.2) }
    }

    #[allow(clippy::wrong_self_convention)]
    #[deprecated(note = "use `into_rect`")]
    #[inline]
    pub const fn to_rect(self) -> (A, A) { self.into_rect() }
}

#[inline] pub const fn from_rect<S: Sign<A>, A>(re: A, im: A) -> Complex<A, S> { Complex::<A, S>::from_rect(re, im) }

impl<S: Sign<A>, A: Clone> Clone for Complex<A, S> {
    #[inline] fn clone(&self) -> Self { Complex(PhantomData, self.1.clone(), self.2.clone()) }
}

impl<S: Sign<A>, A: Copy> Copy for Complex<A, S> {}

impl<S: Sign<A>, A: Zero> Zero for Complex<A, S> {
    const zero: Self = from_rect(A::zero, A::zero);
}

impl<S: Sign<A>, A: Zero + One> One for Complex<A, S> {
    const one: Self = from_rect(A::one, A::zero);
}

impl<S: Sign<A>, A: PartialEq> PartialEq for Complex<A, S> {
    #[inline] fn eq(&self, &Complex(_, ref c, ref d): &Self) -> bool {
        let &Complex(_, ref a, ref b) = self;
        (a, b) == (c, d)
    }
}

impl<S: Sign<A>, A: Eq> Eq for Complex<A, S> {}

impl<S: Sign<A>, A: Copy> From<A> for Complex<A, S> where Z0: Sign<A> {
    fn from(x: A) -> Self { from_rect(x, Z0::sign(x)) }
}

pub trait Conjugable {
    fn conjugate(self) -> Self;
}

impl<S: Sign<A>, A: Add<Output = A> + Neg<Output = A> + Conjugable> Conjugable for Complex<A, S> {
    #[inline] fn conjugate(self) -> Self {
        let Complex(_, a, b) = self;
        Complex(PhantomData, a.conjugate(), b.neg())
    }
}

macro_rules! impl_Conjugable_id {
    ($t: ty) => (impl Conjugable for $t { fn conjugate(self) -> Self { self } });
    ($($t: ty),*) => ($(impl_Conjugable_id!($t);)*);
}
impl_Conjugable_id!((), f32, f64, isize, i8, i16, i32, i64);

impl<S: Sign<A>, A: Add<Output = A>> Add for Complex<A, S> {
    type Output = Self;
    #[inline] fn add(self, Complex(_, c, d): Self) -> Self {
        let Complex(_, a, b) = self;
        Complex(PhantomData, a+c, b+d)
    }
}

impl<S: Sign<A>, A: Sub<Output = A>> Sub for Complex<A, S> {
    type Output = Self;
    #[inline] fn sub(self, Complex(_, c, d): Self) -> Self {
        let Complex(_, a, b) = self;
        Complex(PhantomData, a-c, b-d)
    }
}

impl<S: Sign<A>, A: Neg<Output = A>> Neg for Complex<A, S> {
    type Output = Self;
    #[inline] fn neg(self) -> Self {
        let Complex(_, a, b) = self;
        Complex(PhantomData, -a, -b)
    }
}

impl<S: Sign<A>, A: Copy + Add<Output = A> + Conjugable + Mul<Output = A>> Mul for Complex<A, S> {
    type Output = Self;
    #[inline] fn mul(self, Complex(_, c, d): Self) -> Self {
        let Complex(_, a, b) = self;
        Complex(PhantomData, a*c+S::sign(d.conjugate()*b), d*a+b*c.conjugate())
    }
}

impl<S: Sign<A>, A: Copy + Add<Output = A> + Neg<Output = A> + Conjugable + Mul<Output = A> + Div<Output = A>> Div for Complex<A, S> {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, other: Self) -> Self {
        let Complex(_, a, b) =  self*other.conjugate();
        let Complex(_, c, _) = other*other.conjugate();
        Complex(PhantomData, a/c, b/c)
    }
}

/// Wraps a type to make it opaque to conjugation, i.e. `SelfConjugate(a).conjugate() = SelfConjugate(a)`.
///
/// It can be used to construct higher-order hypercomplex numbers, for example: the dual quaternion type over `A` is `Dual<SelfConjugate<Quaternion<A>>>`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SelfConjugate<A>(pub A);

impl<A> Conjugable for SelfConjugate<A> {
    #[inline]
    fn conjugate(self) -> Self { self }
}

#[cfg(test)] mod tests {
    use typenum::consts::P1;
    use typenum::int::Z0;

    use super::*;

    #[test] fn complex_basis() {
        type T = Complex<isize>;
        let i: T = from_rect(0, 1);
        assert_eq!(from_rect(-1, 0), i*i);
    }

    #[test] fn split_complex_basis() {
        type T = Complex<isize, P1>;
        let i: T = from_rect(0, 1);
        assert_eq!(from_rect( 1, 0), i*i);
    }

    #[test] fn dual_basis() {
        type T = Complex<isize, Z0>;
        let i: T = from_rect(0, 1);
        assert_eq!(from_rect(0, 0), i*i);
    }

    #[test] fn quaternion_basis() {
        type T = Complex<Complex<isize>>;
        let one = from_rect(from_rect(1, 0), from_rect(0, 0));
        let i: T = from_rect(from_rect(0, 1), from_rect(0, 0));
        let j: T = from_rect(from_rect(0, 0), from_rect(1, 0));
        let k: T = from_rect(from_rect(0, 0), from_rect(0, 1));
        assert_eq!((i*j, j*k, k*i, k*j, j*i, i*k,  i*i,  j*j,  k*k),
                   ( k,   i,   j,  -i,  -k,  -j,  -one, -one, -one));
    }
}
