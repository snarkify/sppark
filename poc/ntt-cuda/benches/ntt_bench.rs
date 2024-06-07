use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use ntt_cuda::{NTTInputOutputOrder, NTT, iNTT};
use rand::distributions::{Distribution, Standard};
use rand::random;
use rand::thread_rng;

const DEFAULT_GPU: usize = 0;

fn random_fr_u64() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn random_fr_u32() -> u32 {
    let fr: u32 = random();
    fr % 0x78000001
}

#[cfg(feature = "gl64")]
fn gl64_bench_ntt(c: &mut Criterion) {
    let mut group = c.benchmark_group("NTT");
    let log_sizes = &[14, 16, 18];
    for &lg_domain_size in log_sizes {
        let domain_size = 1 << lg_domain_size;

        let v: Vec<u64> = (0..domain_size).map(|_| random_fr_u64()).collect();
        let mut vtest1 = v.clone();
        let mut vtest2 = v.clone();

        group.bench_with_input(
            BenchmarkId::new("gl64_NN", domain_size),
            &domain_size,
            |b, &_size| {
                b.iter(|| {
                    NTT(DEFAULT_GPU, &mut vtest1, NTTInputOutputOrder::NN);
                });
            },
        );
    }

    group.finish();

}

#[cfg(feature = "bb31")]
fn bb31_bench_ntt(c: &mut Criterion) {
    let mut group = c.benchmark_group("NTT");
    let log_sizes = &[14, 16, 18];
    for &lg_domain_size in log_sizes {
        let domain_size = 1 << lg_domain_size;

        let v: Vec<u32> = (0..domain_size).map(|_| random_fr_u32()).collect();
        let mut vtest1 = v.clone();
        let mut vtest2 = v.clone();

        group.bench_with_input(
            BenchmarkId::new("BB31_NN", domain_size),
            &domain_size,
            |b, &_size| {
                b.iter(|| {
                    NTT(DEFAULT_GPU, &mut vtest1, NTTInputOutputOrder::NN);
                });
            },
        );
    }

    group.finish();

}

fn bench_ntt(c: &mut Criterion) {
    #[cfg(feature = "bb31")]
    bb31_bench_ntt(c);
    #[cfg(feature = "gl64")]
    gl64_bench_ntt(c);
}

criterion_group!(benches, bench_ntt);
criterion_main!(benches);
