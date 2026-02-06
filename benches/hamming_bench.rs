//! Benchmarks for HDR Hamming operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use holograph::{
    BitpackedVector, HdrCascade, VectorField, Resonator,
    hamming::{hamming_distance_scalar, StackedPopcount, Belichtung, HammingEngine},
    VECTOR_BITS,
};

fn random_vectors(count: usize, seed_offset: u64) -> Vec<BitpackedVector> {
    (0..count)
        .map(|i| BitpackedVector::random(i as u64 + seed_offset))
        .collect()
}

/// Benchmark basic Hamming distance
fn bench_hamming_distance(c: &mut Criterion) {
    let a = BitpackedVector::random(1);
    let b = BitpackedVector::random(2);

    let mut group = c.benchmark_group("hamming_distance");
    group.throughput(Throughput::Elements(1));

    group.bench_function("scalar", |bencher| {
        bencher.iter(|| {
            hamming_distance_scalar(black_box(&a), black_box(&b))
        });
    });

    group.finish();
}

/// Benchmark stacked popcount
fn bench_stacked_popcount(c: &mut Criterion) {
    let a = BitpackedVector::random(1);
    let b = BitpackedVector::random(2);

    let mut group = c.benchmark_group("stacked_popcount");
    group.throughput(Throughput::Elements(1));

    group.bench_function("full", |bencher| {
        bencher.iter(|| {
            StackedPopcount::compute(black_box(&a), black_box(&b))
        });
    });

    group.bench_function("with_threshold_pass", |bencher| {
        bencher.iter(|| {
            StackedPopcount::compute_with_threshold(black_box(&a), black_box(&b), 10000)
        });
    });

    group.bench_function("with_threshold_fail", |bencher| {
        bencher.iter(|| {
            StackedPopcount::compute_with_threshold(black_box(&a), black_box(&b), 100)
        });
    });

    group.finish();
}

/// Benchmark Belichtungsmesser (quick exposure meter)
fn bench_belichtung(c: &mut Criterion) {
    let a = BitpackedVector::random(1);
    let b = BitpackedVector::random(2);

    c.bench_function("belichtung_meter", |bencher| {
        bencher.iter(|| {
            Belichtung::meter(black_box(&a), black_box(&b))
        });
    });
}

/// Benchmark bind/unbind operations
fn bench_binding(c: &mut Criterion) {
    let a = BitpackedVector::random(1);
    let b = BitpackedVector::random(2);

    let mut group = c.benchmark_group("binding");

    group.bench_function("bind", |bencher| {
        bencher.iter(|| {
            black_box(&a).xor(black_box(&b))
        });
    });

    let bound = a.xor(&b);
    group.bench_function("unbind", |bencher| {
        bencher.iter(|| {
            black_box(&bound).xor(black_box(&b))
        });
    });

    let c_vec = BitpackedVector::random(3);
    group.bench_function("bind3", |bencher| {
        bencher.iter(|| {
            black_box(&a).xor(black_box(&b)).xor(black_box(&c_vec))
        });
    });

    group.finish();
}

/// Benchmark bundling
fn bench_bundle(c: &mut Criterion) {
    let vecs_3: Vec<_> = (0..3).map(|i| BitpackedVector::random(i)).collect();
    let vecs_7: Vec<_> = (0..7).map(|i| BitpackedVector::random(i)).collect();
    let vecs_16: Vec<_> = (0..16).map(|i| BitpackedVector::random(i)).collect();

    let mut group = c.benchmark_group("bundle");

    group.bench_function("3_vectors", |bencher| {
        let refs: Vec<_> = vecs_3.iter().collect();
        bencher.iter(|| {
            BitpackedVector::bundle(black_box(&refs))
        });
    });

    group.bench_function("7_vectors", |bencher| {
        let refs: Vec<_> = vecs_7.iter().collect();
        bencher.iter(|| {
            BitpackedVector::bundle(black_box(&refs))
        });
    });

    group.bench_function("16_vectors", |bencher| {
        let refs: Vec<_> = vecs_16.iter().collect();
        bencher.iter(|| {
            BitpackedVector::bundle(black_box(&refs))
        });
    });

    group.finish();
}

/// Benchmark HDR cascade search
fn bench_cascade_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("cascade_search");

    for size in [1000, 10000, 100000] {
        let mut cascade = HdrCascade::with_capacity(size);
        let vectors = random_vectors(size, 100);

        for v in &vectors {
            cascade.add(v.clone());
        }

        let query = BitpackedVector::random(150);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("k10", size),
            &(cascade, query),
            |bencher, (cascade, query)| {
                bencher.iter(|| {
                    cascade.search(black_box(query), 10)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch Hamming distance
fn bench_batch_hamming(c: &mut Criterion) {
    let engine = HammingEngine::new();
    let query = BitpackedVector::random(1);
    let candidates: Vec<_> = (0..1000).map(|i| BitpackedVector::random(i + 100)).collect();

    c.bench_function("batch_1000_distances", |bencher| {
        bencher.iter(|| {
            engine.batch_distances(black_box(&query), black_box(&candidates))
        });
    });
}

/// Benchmark KNN search
fn bench_knn(c: &mut Criterion) {
    let engine = HammingEngine::new();
    let query = BitpackedVector::random(1);
    let candidates: Vec<_> = (0..10000).map(|i| BitpackedVector::random(i + 100)).collect();

    let mut group = c.benchmark_group("knn");
    group.throughput(Throughput::Elements(10000));

    for k in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("k", k),
            &k,
            |bencher, &k| {
                bencher.iter(|| {
                    engine.knn(black_box(&query), black_box(&candidates), k)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark resonator
fn bench_resonator(c: &mut Criterion) {
    let mut resonator = Resonator::with_capacity(1000);
    resonator.set_threshold(VECTOR_BITS as u32 / 2);

    for i in 0..1000 {
        resonator.add(BitpackedVector::random(i + 100));
    }

    let query = BitpackedVector::random(500); // Should match entry 400

    c.bench_function("resonator_1000", |bencher| {
        bencher.iter(|| {
            resonator.resonate(black_box(&query))
        });
    });
}

/// Benchmark vector creation
fn bench_vector_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_creation");

    group.bench_function("zero", |bencher| {
        bencher.iter(|| {
            BitpackedVector::zero()
        });
    });

    group.bench_function("random", |bencher| {
        let mut seed = 0u64;
        bencher.iter(|| {
            seed += 1;
            BitpackedVector::random(black_box(seed))
        });
    });

    let data = b"Hello, world! This is test data for hashing.";
    group.bench_function("from_hash", |bencher| {
        bencher.iter(|| {
            BitpackedVector::from_hash(black_box(data))
        });
    });

    group.finish();
}

/// Benchmark memory operations
fn bench_memory(c: &mut Criterion) {
    let v = BitpackedVector::random(1);

    let mut group = c.benchmark_group("memory");

    group.bench_function("clone", |bencher| {
        bencher.iter(|| {
            black_box(&v).clone()
        });
    });

    group.bench_function("to_bytes", |bencher| {
        bencher.iter(|| {
            black_box(&v).to_bytes()
        });
    });

    let bytes = v.to_bytes();
    group.bench_function("from_bytes", |bencher| {
        bencher.iter(|| {
            BitpackedVector::from_bytes(black_box(&bytes))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hamming_distance,
    bench_stacked_popcount,
    bench_belichtung,
    bench_binding,
    bench_bundle,
    bench_cascade_search,
    bench_batch_hamming,
    bench_knn,
    bench_resonator,
    bench_vector_creation,
    bench_memory,
);

criterion_main!(benches);
