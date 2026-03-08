import numpy as np
from GetUEFC import UEFC
from DS_scan_ARS import scan_ARS
from DS_report_opt_obj import report_opt_obj


def make_aircraft(dbmax, CLdes):
    aircraft = UEFC()

    aircraft.mpay_g = 250
    aircraft.dihedral = 10.0
    aircraft.Sh = 0.04
    aircraft.Sv = 0.03
    aircraft.l_AR = 1.63
    aircraft.e0 = 1.0
    aircraft.rhofoam = 32.0
    aircraft.Efoam = 19.3e6

    aircraft.dbmax = dbmax
    aircraft.CLdes = CLdes

    return aircraft


def print_summary_table(results, title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'lambda':>8} {'tau':>8} {'Vopt (m/s)':>12} {'ARopt':>10} {'Sopt (m^2)':>12}")
    print("-" * 78)

    for r in results:
        if r["Vopt"] > 0:
            print(f"{r['lambda']:8.2f} {r['tau']:8.2f} {r['Vopt']:12.4f} {r['ARopt']:10.4f} {r['Sopt']:12.5f}")
        else:
            print(f"{r['lambda']:8.2f} {r['tau']:8.2f} {'FAILED':>12} {'---':>10} {'---':>12}")

    print("=" * 78)


def sweep_lambda_tau(dbmax, CLdes,
                     lam_vals=np.linspace(0.5, 1.0, 6),
                     tau_vals=np.linspace(0.08, 0.12, 5),
                     AR_start=4.0, AR_end=9.0,
                     S_start=0.125, S_end=0.185,
                     num_division=11):

    results = []

    total_cases = len(lam_vals) * len(tau_vals)
    case_num = 0

    for lam in lam_vals:
        for tau in tau_vals:
            case_num += 1
            print(f"\nRunning case {case_num}/{total_cases}: lambda={lam:.2f}, tau={tau:.2f}")

            aircraft = make_aircraft(dbmax, CLdes)
            aircraft.taper = lam
            aircraft.tau = tau

            Vopt, ARopt, Sopt = scan_ARS(
                aircraft,
                AR_start=AR_start, AR_end=AR_end,
                S_start=S_start, S_end=S_end,
                num_division=num_division,
                show_plots=False
            )

            results.append({
                "lambda": lam,
                "tau": tau,
                "Vopt": Vopt,
                "ARopt": ARopt,
                "Sopt": Sopt
            })

            if Vopt > 0:
                print(f"Done: Vopt={Vopt:.4f} m/s, ARopt={ARopt:.4f}, Sopt={Sopt:.5f}")
            else:
                print("Done: no feasible solution found")

    feasible = [r for r in results if r["Vopt"] > 0]

    if len(feasible) == 0:
        best = None
        worst = None
    else:
        best = max(feasible, key=lambda x: x["Vopt"])
        worst = min(feasible, key=lambda x: x["Vopt"])

    return results, best, worst


def show_case(case, dbmax, CLdes, label,
              AR_start=4.0, AR_end=9.0,
              S_start=0.125, S_end=0.185,
              num_division=41):

    if case is None:
        print(f"\n{label}: No feasible case found.")
        return

    print(f"\n{'='*30}")
    print(label)
    print(f"{'='*30}")
    print(f"lambda = {case['lambda']:.2f}")
    print(f"tau    = {case['tau']:.2f}")
    print(f"Vopt   = {case['Vopt']:.4f} m/s")
    print(f"ARopt  = {case['ARopt']:.4f}")
    print(f"Sopt   = {case['Sopt']:.5f} m^2")

    aircraft = make_aircraft(dbmax, CLdes)
    aircraft.taper = case["lambda"]
    aircraft.tau = case["tau"]

    Vref, ARref, Sref = scan_ARS(
        aircraft,
        AR_start=AR_start, AR_end=AR_end,
        S_start=S_start, S_end=S_end,
        num_division=num_division,
        show_plots=True
    )

    print("\nRefined optimum from plotting scan:")
    print(f"Vopt_refined = {Vref:.4f} m/s")
    print(f"ARopt_refined = {ARref:.4f}")
    print(f"Sopt_refined = {Sref:.5f} m^2")

    print(f"\nDetailed report for {label.lower()}:")
    report_opt_obj(aircraft, ARref, Sref)


# =========================
# DS.4
# =========================
results4, best4, worst4 = sweep_lambda_tau(
    dbmax=0.08,
    CLdes=0.75,
    num_division=11
)

print_summary_table(results4, "DS.4 SUMMARY TABLE  (dbmax = 0.08, CLdes = 0.75)")

show_case(best4, 0.08, 0.75, "DS.4 BEST CASE",  num_division=41)
show_case(worst4, 0.08, 0.75, "DS.4 WORST CASE", num_division=41)


# =========================
# DS.5
# =========================
results5, best5, worst5 = sweep_lambda_tau(
    dbmax=0.10,
    CLdes=0.90,
    num_division=11
)

print_summary_table(results5, "DS.5 SUMMARY TABLE  (dbmax = 0.10, CLdes = 0.90)")

show_case(best5, 0.10, 0.90, "DS.5 BEST CASE",  num_division=41)
show_case(worst5, 0.10, 0.90, "DS.5 WORST CASE", num_division=41)
