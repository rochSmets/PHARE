#ifndef PHARE_CORE_NUMERICS_OHM_VACUUM_AMPERE_RELAX_HPP
#define PHARE_CORE_NUMERICS_OHM_VACUUM_AMPERE_RELAX_HPP

#include <cmath>

namespace PHARE::core
{

struct VacuumAmpereRelaxInfo
{
    double const n0;    // transition density
    double const p;     // falloff power of g(n), p >= 2 recommended
    double const c_eff; // effective/reduced speed of light, in v_A = 1 units
};


// scalar, grid-independent implementation of the exact exponential-integrator
// solution of  dE/dt = lambda(n) * (E_Ohm - E) + c_eff^2 * curl(B)
// with lambda(n) = g(n) / dt  and  g(n) = (n / n0)^p, so that g is dt-independent.
// See doc/design/vacuum_ampere_ohm_blend.md for the derivation.
class VacuumAmpereRelax : public VacuumAmpereRelaxInfo
{
public:
    explicit VacuumAmpereRelax(VacuumAmpereRelaxInfo const& info)
        : VacuumAmpereRelaxInfo{info}
    {
    }

    double g(double n) const
    {
        if (n <= 0.0)
            return 0.0;
        return std::pow(n / n0, p);
    }

    static double w(double g_) { return std::exp(-g_); }

    static double h(double g_)
    {
        if (g_ < 1e-12) // guards the 0/0 at g==0 ; expm1 is otherwise well conditioned
            return 1.0;
        return -std::expm1(-g_) / g_;
    }

    double operator()(double Eold, double Eohm, double curlB, double n, double dt) const
    {
        auto const g_ = g(n);
        auto const w_ = w(g_);
        auto const h_ = h(g_);
        return w_ * Eold + (1.0 - w_) * Eohm + dt * h_ * c_eff * c_eff * curlB;
    }
};

} // namespace PHARE::core

#endif
