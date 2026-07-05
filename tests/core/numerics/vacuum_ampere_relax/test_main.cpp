// Standalone, grid-independent validation of the exact exponential-integrator
// closed form used to blend Ohm's law and vacuum Ampere's law for E.
// See doc/design/vacuum_ampere_ohm_blend.md for the derivation this checks.

#include "core/numerics/ohm/vacuum_ampere_relax.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <vector>

using namespace PHARE::core;


namespace
{
// brute-force reference: explicit forward-Euler integration of the same ODE
//   dE/dt = lambda(n) * (Eohm - E) + c_eff^2 * curlB ,  lambda(n) = g(n)/dt
// used only to independently validate the closed-form formula in
// VacuumAmpereRelax, not as an implementation to keep.
double bruteForceEuler(VacuumAmpereRelax const& relax, double Eold, double Eohm, double curlB,
                       double n, double dt, std::size_t M)
{
    auto const g   = relax.g(n);
    auto const lam = g / dt;
    auto const dtf = dt / static_cast<double>(M);

    auto E = Eold;
    for (std::size_t i = 0; i < M; ++i)
    {
        auto const dE = lam * (Eohm - E) + relax.c_eff * relax.c_eff * curlB;
        E += dtf * dE;
    }
    return E;
}
} // namespace



TEST(VacuumAmpereRelax, VacuumLimitIsExactEvenWithHugeFinitePoison)
{
    VacuumAmpereRelax const relax{{/*n0=*/1.0, /*p=*/2.0, /*c_eff=*/50.0}};

    double const Eold  = 3.0;
    double const curlB = 0.7;
    double const dt    = 0.05;
    double const expected = Eold + dt * relax.c_eff * relax.c_eff * curlB;

    for (double poisonedEohm : {1e6, -1e12, 1e300})
    {
        auto const got = relax(Eold, poisonedEohm, curlB, /*n=*/0.0, dt);
        EXPECT_DOUBLE_EQ(got, expected);
    }
}



TEST(VacuumAmpereRelax, VacuumLimitDoesNotSurviveNaNOrInfPoisoning)
{
    // documents a real hazard (see design doc section 4.1 addendum): the weight
    // (1-w) is exactly 0 at n=0, but 0 * NaN == NaN and 0 * Inf == NaN in IEEE
    // arithmetic. this is why n MUST be floored inside Ohm's law's singular
    // terms before it ever reaches this blend -- the weight alone does not
    // protect against a genuine x/0 upstream.
    VacuumAmpereRelax const relax{{1.0, 2.0, 50.0}};

    auto const nan = std::numeric_limits<double>::quiet_NaN();
    auto const inf = std::numeric_limits<double>::infinity();

    EXPECT_TRUE(std::isnan(relax(3.0, nan, 0.7, 0.0, 0.05)));
    EXPECT_TRUE(std::isnan(relax(3.0, inf, 0.7, 0.0, 0.05)));
}



TEST(VacuumAmpereRelax, DenseLimitResidualScalesAsInverseG)
{
    double const n0    = 1.0;
    double const p     = 2.0;
    double const c_eff = 50.0;
    VacuumAmpereRelax const relax{{n0, p, c_eff}};

    double const Eold  = 1.0;
    double const Eohm  = 42.0;
    double const curlB = 0.3;
    double const dt    = 0.01;

    // n chosen so that g = (n/n0)^p is exactly 100 and 1000 respectively
    double const n_g100  = 10.0;
    double const n_g1000 = std::sqrt(1000.0);

    ASSERT_NEAR(relax.g(n_g100), 100.0, 1e-9);
    ASSERT_NEAR(relax.g(n_g1000), 1000.0, 1e-9);

    auto const E_g100  = relax(Eold, Eohm, curlB, n_g100, dt);
    auto const E_g1000 = relax(Eold, Eohm, curlB, n_g1000, dt);

    auto const dev_g100  = std::abs(E_g100 - Eohm);
    auto const dev_g1000 = std::abs(E_g1000 - Eohm);

    // deviation from E_Ohm ~ dt * c_eff^2 * curlB / g for large g, so a 10x
    // increase in g should give a ~10x decrease in the deviation.
    EXPECT_NEAR(dev_g100 / dev_g1000, 10.0, 0.1);
}



TEST(VacuumAmpereRelax, ClosedFormMatchesBruteForceEulerIntegration)
{
    VacuumAmpereRelax const relax{{/*n0=*/1.0, /*p=*/2.0, /*c_eff=*/50.0}};

    struct Case
    {
        double Eold, Eohm, curlB, n, dt;
    };

    std::vector<Case> const cases{
        {1.0, 2.0, 0.5, 0.1, 0.01},
        {0.5, -3.0, 1.0, 0.3, 0.1},
        {2.0, 5.0, -0.2, 1.0, 0.01},
        {1.0, 100.0, 0.1, 3.0, 0.01},
        {1.0, 1000.0, 0.1, 10.0, 0.01},
    };

    for (auto const& c : cases)
    {
        auto const closed = relax(c.Eold, c.Eohm, c.curlB, c.n, c.dt);
        auto const brute  = bruteForceEuler(relax, c.Eold, c.Eohm, c.curlB, c.n, c.dt, 200000);

        auto const scale = std::max(1.0, std::abs(closed));
        EXPECT_NEAR(closed, brute, 1e-4 * scale)
            << "n=" << c.n << " g=" << relax.g(c.n);
    }
}



TEST(VacuumAmpereRelax, RecommendedFalloffSuppressesOhmSingularityAsDensityVanishes)
{
    // this encodes the pitfall documented in the design doc section 4.2:
    // lambda(n) must vanish faster than n as n->0, otherwise the (1-w)*E_Ohm
    // term does not suppress the 1/n Hall/pressure singularities already
    // present in Ohm's law. compares the recommended g(n) = (n/n0)^p, p=2,
    // against the physically-tempting-but-wrong lambda ~ omega_pe ~ sqrt(n).
    double const n0        = 1.0;
    double const dt        = 0.01;
    double const C         = 1.0; // mimics a Hall-term-like E_Ohm(n) = C/n singularity

    VacuumAmpereRelax const recommended{{n0, /*p=*/2.0, /*c_eff=*/50.0}};

    auto const wrongW = [&](double n) { return std::exp(-std::sqrt(n) * dt); };

    double previousRecommended = std::numeric_limits<double>::infinity();
    for (double n : {1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8})
    {
        double const Eohm = C / n;

        auto const recommendedContribution = (1.0 - recommended.w(recommended.g(n))) * Eohm;
        auto const wrongContribution        = (1.0 - wrongW(n)) * Eohm;

        // recommended: leaked contribution shrinks monotonically to zero
        EXPECT_LT(recommendedContribution, previousRecommended);
        previousRecommended = recommendedContribution;

        // wrong (lambda ~ sqrt(n)): leaked contribution grows without bound,
        // ~ dt*C/sqrt(n) asymptotically; check against half that as a safe
        // lower bound.
        EXPECT_GT(wrongContribution, 0.5 * dt / std::sqrt(n));
    }
    EXPECT_LT(previousRecommended, 1e-6);
}
