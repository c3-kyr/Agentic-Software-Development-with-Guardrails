/**
 * Simulation Engine — Pre-built scenarios & random generator
 * Each scenario models an architectural decision with ground-truth signals.
 */

// Seeded PRNG for reproducible random scenarios
function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Generate a random embedding vector of given dimension
function randomEmbedding(dim, rng) {
  const v = [];
  for (let i = 0; i < dim; i++) {
    v.push(rng() * 2 - 1); // range [-1, 1]
  }
  // Normalize
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / norm);
}

// Generate a perturbed copy of an embedding
function perturbEmbedding(base, noise, rng) {
  const v = base.map((x) => x + (rng() * 2 - 1) * noise);
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / norm);
}

const CONFLICT_POOL = [
  "Scaling strategy unclear",
  "Database normalization incomplete",
  "Schema migration risk",
  "Single point of failure",
  "Vendor lock-in concerns",
  "Data consistency challenges",
  "Latency requirements unmet",
  "Cost projection uncertain",
  "Security model incomplete",
  "Team expertise mismatch",
  "Integration complexity high",
  "Recovery strategy undefined",
  "Compliance requirements unclear",
  "Performance bottleneck risk",
  "Monitoring gaps identified",
];

const SEVERITY_MAP = {
  "Scaling strategy unclear": 0.8,
  "Database normalization incomplete": 0.6,
  "Schema migration risk": 0.9,
  "Single point of failure": 0.95,
  "Vendor lock-in concerns": 0.7,
  "Data consistency challenges": 0.85,
  "Latency requirements unmet": 0.75,
  "Cost projection uncertain": 0.5,
  "Security model incomplete": 0.9,
  "Team expertise mismatch": 0.4,
  "Integration complexity high": 0.65,
  "Recovery strategy undefined": 0.85,
  "Compliance requirements unclear": 0.7,
  "Performance bottleneck risk": 0.8,
  "Monitoring gaps identified": 0.55,
};

/** Pre-built architectural decision scenarios */
export const PRESET_SCENARIOS = [
  {
    id: "db-user-service",
    label: "Database for User Service",
    category: "Database",
    options: ["PostgreSQL", "MongoDB"],
    impact: 0.8,
    irreversibility: 0.7,
    propagation: 0.9,
    confidenceDistribution: { PostgreSQL: 0.55, MongoDB: 0.45 },
    reflectionAgreement: 0.62,
    reflectionConflicts: ["Schema migration risk", "Data consistency challenges"],
    groundTruthNeedsHuman: true,
  },
  {
    id: "cache-strategy",
    label: "Caching Layer Choice",
    category: "Infrastructure",
    options: ["Redis", "Memcached", "Application-level"],
    impact: 0.5,
    irreversibility: 0.4,
    propagation: 0.6,
    confidenceDistribution: { Redis: 0.7, Memcached: 0.2, "Application-level": 0.1 },
    reflectionAgreement: 0.85,
    reflectionConflicts: ["Cost projection uncertain"],
    groundTruthNeedsHuman: false,
  },
  {
    id: "arch-pattern",
    label: "Monolith vs Microservices",
    category: "Architecture",
    options: ["Monolith", "Microservices"],
    impact: 0.9,
    irreversibility: 0.9,
    propagation: 1.0,
    confidenceDistribution: { Monolith: 0.48, Microservices: 0.52 },
    reflectionAgreement: 0.45,
    reflectionConflicts: [
      "Scaling strategy unclear",
      "Team expertise mismatch",
      "Integration complexity high",
    ],
    groundTruthNeedsHuman: true,
  },
  {
    id: "auth-provider",
    label: "Authentication Provider",
    category: "Security",
    options: ["Auth0", "Firebase Auth", "Custom OAuth"],
    impact: 0.7,
    irreversibility: 0.6,
    propagation: 0.8,
    confidenceDistribution: { Auth0: 0.5, "Firebase Auth": 0.35, "Custom OAuth": 0.15 },
    reflectionAgreement: 0.72,
    reflectionConflicts: ["Vendor lock-in concerns", "Security model incomplete"],
    groundTruthNeedsHuman: true,
  },
  {
    id: "api-protocol",
    label: "API Protocol Selection",
    category: "API",
    options: ["REST", "GraphQL", "gRPC"],
    impact: 0.6,
    irreversibility: 0.5,
    propagation: 0.7,
    confidenceDistribution: { REST: 0.6, GraphQL: 0.3, gRPC: 0.1 },
    reflectionAgreement: 0.78,
    reflectionConflicts: ["Integration complexity high"],
    groundTruthNeedsHuman: false,
  },
  {
    id: "msg-queue",
    label: "Message Queue System",
    category: "Infrastructure",
    options: ["RabbitMQ", "Kafka", "SQS"],
    impact: 0.7,
    irreversibility: 0.6,
    propagation: 0.8,
    confidenceDistribution: { RabbitMQ: 0.35, Kafka: 0.4, SQS: 0.25 },
    reflectionAgreement: 0.58,
    reflectionConflicts: ["Scaling strategy unclear", "Vendor lock-in concerns"],
    groundTruthNeedsHuman: true,
  },
  {
    id: "frontend-framework",
    label: "Frontend Framework",
    category: "Frontend",
    options: ["React", "Vue", "Svelte"],
    impact: 0.4,
    irreversibility: 0.5,
    propagation: 0.4,
    confidenceDistribution: { React: 0.65, Vue: 0.25, Svelte: 0.1 },
    reflectionAgreement: 0.88,
    reflectionConflicts: [],
    groundTruthNeedsHuman: false,
  },
  {
    id: "deploy-strategy",
    label: "Deployment Strategy",
    category: "DevOps",
    options: ["Kubernetes", "Serverless", "VM-based"],
    impact: 0.6,
    irreversibility: 0.7,
    propagation: 0.8,
    confidenceDistribution: { Kubernetes: 0.45, Serverless: 0.35, "VM-based": 0.2 },
    reflectionAgreement: 0.65,
    reflectionConflicts: ["Cost projection uncertain", "Team expertise mismatch"],
    groundTruthNeedsHuman: true,
  },
  {
    id: "logging-stack",
    label: "Logging & Monitoring Stack",
    category: "Observability",
    options: ["ELK Stack", "Datadog", "Prometheus+Grafana"],
    impact: 0.3,
    irreversibility: 0.3,
    propagation: 0.5,
    confidenceDistribution: { "ELK Stack": 0.35, Datadog: 0.4, "Prometheus+Grafana": 0.25 },
    reflectionAgreement: 0.82,
    reflectionConflicts: ["Monitoring gaps identified"],
    groundTruthNeedsHuman: false,
  },
  {
    id: "state-mgmt",
    label: "State Management Pattern",
    category: "Frontend",
    options: ["Redux", "Zustand", "Context API"],
    impact: 0.3,
    irreversibility: 0.3,
    propagation: 0.3,
    confidenceDistribution: { Redux: 0.3, Zustand: 0.45, "Context API": 0.25 },
    reflectionAgreement: 0.9,
    reflectionConflicts: [],
    groundTruthNeedsHuman: false,
  },
  {
    id: "data-pipeline",
    label: "Data Pipeline Architecture",
    category: "Data",
    options: ["Batch ETL", "Stream Processing", "Lambda Architecture"],
    impact: 0.8,
    irreversibility: 0.8,
    propagation: 0.9,
    confidenceDistribution: { "Batch ETL": 0.3, "Stream Processing": 0.4, "Lambda Architecture": 0.3 },
    reflectionAgreement: 0.5,
    reflectionConflicts: [
      "Scaling strategy unclear",
      "Performance bottleneck risk",
      "Recovery strategy undefined",
    ],
    groundTruthNeedsHuman: true,
  },
  {
    id: "search-engine",
    label: "Search Engine Selection",
    category: "Infrastructure",
    options: ["Elasticsearch", "Algolia", "PostgreSQL FTS"],
    impact: 0.5,
    irreversibility: 0.4,
    propagation: 0.5,
    confidenceDistribution: { Elasticsearch: 0.5, Algolia: 0.35, "PostgreSQL FTS": 0.15 },
    reflectionAgreement: 0.75,
    reflectionConflicts: ["Cost projection uncertain"],
    groundTruthNeedsHuman: false,
  },
  {
    id: "ci-cd-tool",
    label: "CI/CD Pipeline Tool",
    category: "DevOps",
    options: ["GitHub Actions", "Jenkins", "GitLab CI"],
    impact: 0.3,
    irreversibility: 0.2,
    propagation: 0.3,
    confidenceDistribution: { "GitHub Actions": 0.6, Jenkins: 0.25, "GitLab CI": 0.15 },
    reflectionAgreement: 0.92,
    reflectionConflicts: [],
    groundTruthNeedsHuman: false,
  },
  {
    id: "encryption-at-rest",
    label: "Encryption at Rest Strategy",
    category: "Security",
    options: ["AES-256 Application-level", "Database TDE", "Cloud KMS"],
    impact: 0.9,
    irreversibility: 0.7,
    propagation: 0.6,
    confidenceDistribution: { "AES-256 Application-level": 0.3, "Database TDE": 0.35, "Cloud KMS": 0.35 },
    reflectionAgreement: 0.55,
    reflectionConflicts: ["Security model incomplete", "Compliance requirements unclear"],
    groundTruthNeedsHuman: true,
  },
  {
    id: "file-storage",
    label: "File Storage Solution",
    category: "Infrastructure",
    options: ["S3", "GCS", "Local NFS"],
    impact: 0.4,
    irreversibility: 0.3,
    propagation: 0.4,
    confidenceDistribution: { S3: 0.65, GCS: 0.25, "Local NFS": 0.1 },
    reflectionAgreement: 0.88,
    reflectionConflicts: [],
    groundTruthNeedsHuman: false,
  },
  {
    id: "orm-choice",
    label: "ORM vs Raw SQL",
    category: "Database",
    options: ["Prisma", "TypeORM", "Raw SQL"],
    impact: 0.4,
    irreversibility: 0.4,
    propagation: 0.5,
    confidenceDistribution: { Prisma: 0.5, TypeORM: 0.3, "Raw SQL": 0.2 },
    reflectionAgreement: 0.8,
    reflectionConflicts: ["Performance bottleneck risk"],
    groundTruthNeedsHuman: false,
  },
  {
    id: "session-mgmt",
    label: "Session Management Approach",
    category: "Security",
    options: ["JWT Stateless", "Server-side Sessions", "Hybrid"],
    impact: 0.6,
    irreversibility: 0.5,
    propagation: 0.7,
    confidenceDistribution: { "JWT Stateless": 0.45, "Server-side Sessions": 0.35, Hybrid: 0.2 },
    reflectionAgreement: 0.68,
    reflectionConflicts: ["Security model incomplete", "Scaling strategy unclear"],
    groundTruthNeedsHuman: true,
  },
  {
    id: "testing-strategy",
    label: "Testing Strategy Priority",
    category: "Quality",
    options: ["Unit-heavy", "Integration-heavy", "E2E-heavy"],
    impact: 0.3,
    irreversibility: 0.2,
    propagation: 0.3,
    confidenceDistribution: { "Unit-heavy": 0.5, "Integration-heavy": 0.35, "E2E-heavy": 0.15 },
    reflectionAgreement: 0.85,
    reflectionConflicts: [],
    groundTruthNeedsHuman: false,
  },
  {
    id: "multi-tenancy",
    label: "Multi-Tenancy Architecture",
    category: "Architecture",
    options: ["Shared DB", "Schema-per-tenant", "DB-per-tenant"],
    impact: 0.9,
    irreversibility: 0.95,
    propagation: 1.0,
    confidenceDistribution: { "Shared DB": 0.33, "Schema-per-tenant": 0.34, "DB-per-tenant": 0.33 },
    reflectionAgreement: 0.38,
    reflectionConflicts: [
      "Data consistency challenges",
      "Scaling strategy unclear",
      "Schema migration risk",
      "Single point of failure",
    ],
    groundTruthNeedsHuman: true,
  },
  {
    id: "rate-limiting",
    label: "Rate Limiting Strategy",
    category: "API",
    options: ["Token Bucket", "Sliding Window", "Fixed Window"],
    impact: 0.3,
    irreversibility: 0.2,
    propagation: 0.4,
    confidenceDistribution: { "Token Bucket": 0.5, "Sliding Window": 0.35, "Fixed Window": 0.15 },
    reflectionAgreement: 0.87,
    reflectionConflicts: [],
    groundTruthNeedsHuman: false,
  },
];

// Pre-generate embeddings for preset scenarios
const DIM = 8;
const presetRng = mulberry32(42);
PRESET_SCENARIOS.forEach((s) => {
  // Generate embeddings that reflect scenario uncertainty
  const baseEmb = randomEmbedding(DIM, presetRng);
  const noise = 1 - s.reflectionAgreement; // Higher disagreement → more noise
  s.embeddingVectors = [];
  for (let i = 0; i < 5; i++) {
    s.embeddingVectors.push(perturbEmbedding(baseEmb, noise * 0.8, presetRng));
  }
  // Add conflict severities
  s.conflictSeverities = s.reflectionConflicts.map(
    (c) => SEVERITY_MAP[c] || 0.5
  );
});

/**
 * Generate a random scenario with controllable parameters
 */
export function generateRandomScenario(seed) {
  const rng = mulberry32(seed);

  const categories = [
    "Database", "Infrastructure", "Architecture", "Security",
    "API", "Frontend", "DevOps", "Data", "Quality",
  ];
  const labels = [
    "Storage Tier Selection", "Network Topology", "Event Bus Design",
    "Concurrency Model", "Backup Strategy", "CDN Configuration",
    "Service Mesh Decision", "Feature Flag System", "Rollback Mechanism",
    "Error Handling Pattern", "Config Management", "Secret Rotation",
  ];

  const numOptions = 2 + Math.floor(rng() * 2); // 2-3 options
  const options = [];
  for (let i = 0; i < numOptions; i++) {
    options.push(`Option_${String.fromCharCode(65 + i)}`);
  }

  // Random confidence distribution
  const rawProbs = options.map(() => rng());
  const sumProbs = rawProbs.reduce((a, b) => a + b, 0);
  const conf = {};
  options.forEach((opt, i) => {
    conf[opt] = rawProbs[i] / sumProbs;
  });

  const impact = Math.round(rng() * 10) / 10;
  const irreversibility = Math.round(rng() * 10) / 10;
  const propagation = Math.round(rng() * 10) / 10;
  const agreement = Math.round(rng() * 100) / 100;

  // Pick random conflicts
  const numConflicts = Math.floor(rng() * 4);
  const shuffled = [...CONFLICT_POOL].sort(() => rng() - 0.5);
  const conflicts = shuffled.slice(0, numConflicts);

  const baseEmb = randomEmbedding(DIM, rng);
  const noise = 1 - agreement;
  const embeddings = [];
  for (let i = 0; i < 5; i++) {
    embeddings.push(perturbEmbedding(baseEmb, noise * 0.8, rng));
  }

  // Heuristic: high gravity + low agreement → needs human
  const gravity = impact * irreversibility * propagation;
  const needsHuman = gravity > 0.3 && agreement < 0.7 ? true : gravity > 0.5 ? rng() > 0.5 : false;

  return {
    id: `random-${seed}`,
    label: labels[seed % labels.length] + ` #${seed}`,
    category: categories[Math.floor(rng() * categories.length)],
    options,
    impact,
    irreversibility,
    propagation,
    confidenceDistribution: conf,
    reflectionAgreement: agreement,
    reflectionConflicts: conflicts,
    conflictSeverities: conflicts.map((c) => SEVERITY_MAP[c] || 0.5),
    embeddingVectors: embeddings,
    groundTruthNeedsHuman: needsHuman,
  };
}

/**
 * Generate a batch of random scenarios
 */
export function generateRandomBatch(count, baseSeed = Date.now()) {
  const scenarios = [];
  for (let i = 0; i < count; i++) {
    scenarios.push(generateRandomScenario(baseSeed + i));
  }
  return scenarios;
}
