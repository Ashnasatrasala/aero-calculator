# hybrid_solver.py
import math
import sympy as sp
from pint import UnitRegistry
import regex as re
from typing import Dict, Tuple, Optional, List

ureg = UnitRegistry()
Q_ = ureg.Quantity

# -----------------------
# ISA density helper (simple troposphere model up to 11 km)
# -----------------------
def isa_density(alt_m: float) -> float:
    T0 = 288.15        # K
    P0 = 101325.0      # Pa
    g = 9.80665
    L = 0.0065
    R = 287.058
    if alt_m <= 11000:
        T = T0 - L * alt_m
        P = P0 * (T / T0) ** (g / (R * L))
    else:
        P = P0 * math.exp(-alt_m / 7000.0)
        T = T0 - L * 11000
    rho = P / (R * T)
    return rho

# -----------------------
# Load formulas from text file
# -----------------------
def load_formulas(filepath: str):
    formulas = []
    with open(filepath, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            desc = ""
            if "::" in line:
                left, desc = (p.strip() for p in line.split("::", 1))
            else:
                left = line

            if "=" not in left:
                print(f"[WARN] skipping invalid formula (no '='): {line}")
                continue

            lhs_text, rhs_text = [p.strip() for p in left.split("=", 1)]

            # --- Extract variables using regex ----
            # variable = combination of letters or letters+numbers
            import re
            var_candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", left)

            # remove function names like log
            banned = {"log", "sin", "cos", "tan", "exp"}
            vars_clean = [v for v in var_candidates if v not in banned]

            # Make all of them SymPy symbols
            symbols_dict = {v: sp.Symbol(v) for v in vars_clean}

            # Parse safely using locals=symbols_dict
            try:
                lhs = sp.sympify(lhs_text, locals=symbols_dict)
                rhs = sp.sympify(rhs_text, locals=symbols_dict)
                expr = lhs - rhs
                formulas.append((left, desc, expr))

            except Exception as e:
                print(f"[ERROR] Cannot parse formula: {line}\n  {e}")

    return formulas


# -----------------------
# Extraction heuristics
# -----------------------
VAR_EQ_PATTERN = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([+-]?\d+(\.\d+)?(?:[eE][+-]?\d+)?)(?:\s*([A-Za-z/%\*\^\-0-9·]+(?:\s*/\s*[A-Za-z]+)?))?')
NUMBER_UNIT_PATTERN = re.compile(r'([+-]?\d+(\.\d+)?(?:[eE][+-]?\d+)?)\s*([A-Za-zμμ/·\*\-0-9]+)')

def try_parse_quantity(num_str: str, unit_str: Optional[str]) -> Q_:
    if unit_str:
        unit_str = unit_str.replace('^', '**').replace('·', ' ').strip()
        try:
            return Q_(float(num_str), ureg.parse_units(unit_str))
        except Exception:
            try:
                return ureg.parse_expression(f"{num_str} {unit_str}")
            except Exception:
                raise
    else:
        return Q_(float(num_str))

def extract_values_from_text(situation: str, variables: List[sp.Symbol]) -> Dict[str, Q_]:
    s = situation
    found: Dict[str, Q_] = {}
    var_names = {str(v): v for v in variables}

    # explicit name = value unit
    for m in VAR_EQ_PATTERN.finditer(s):
        name = m.group(1)
        num = m.group(2)
        unit = m.group(4)
        if name in var_names:
            try:
                qty = try_parse_quantity(num, unit)
                found[name] = qty
            except Exception:
                pass

    # token-based lookups like "v 200 m/s"
    tokens = s.split()
    for i, tok in enumerate(tokens):
        tclean = re.sub(r'[^A-Za-z0-9_]', '', tok)
        if tclean in var_names:
            window = " ".join(tokens[i+1:i+4])
            m2 = NUMBER_UNIT_PATTERN.search(window)
            if m2:
                num = m2.group(1)
                unit = m2.group(3)
                try:
                    qty = try_parse_quantity(num, unit)
                    found[tclean] = qty
                except Exception:
                    pass

    # altitude -> rho
    alt_match = re.search(r'\b(altitude|alt|h|height)\b[^\d]{0,10}([+-]?\d+(?:\.\d+)?)(?:\s*(m|km|ft|meter|meters|metre|metres))?', s, flags=re.IGNORECASE)
    if alt_match:
        rawnum = alt_match.group(2)
        unit = alt_match.group(3) or 'm'
        unit = unit.lower().replace('.', '')
        alt_val = float(rawnum)
        if unit in ['ft']:
            alt_m = alt_val * 0.3048
        elif unit in ['km']:
            alt_m = alt_val * 1000.0
        else:
            alt_m = alt_val
        rho_val = isa_density(alt_m)
        if 'rho' in var_names:
            found['rho'] = Q_(rho_val, 'kg/m**3')

    return found

# -----------------------
# Prepare substitutions (convert quantities to SI floats)
# -----------------------
def prepare_subs(expr: sp.Expr, found_quantities: Dict[str, Q_]) -> Dict[sp.Symbol, float]:
    subs = {}
    for sym in expr.free_symbols:
        name = str(sym)
        if name in found_quantities:
            q = found_quantities[name]
            try:
                q_si = q.to_base_units()
                subs[sym] = float(q_si.magnitude)
            except Exception:
                subs[sym] = float(q.magnitude)
    return subs

# -----------------------
# Solve function
# -----------------------
def solve_formula(expr: sp.Expr, target: Optional[str], subs: Dict[sp.Symbol, float]) -> Tuple[Dict[str, float], Dict]:
    syms = sorted(expr.free_symbols, key=lambda x: str(x))
    unknowns = [s for s in syms if s not in subs]
    metadata = {'unknowns': [str(u) for u in unknowns], 'given': {str(k): v for k, v in subs.items()}}
    results = {}
    if target:
        target_sym = sp.Symbol(target)
        sol = sp.solve(expr, target_sym, dict=True)
        if not sol:
            raise ValueError(f"Cannot rearrange formula to solve for {target}")
        sol_expr = sol[0][target_sym]
        val = float(sol_expr.subs(subs).evalf())
        results[target] = val
        metadata['expr_solved'] = str(sol_expr)
        return results, metadata
    else:
        for u in unknowns:
            sol = sp.solve(expr, u, dict=True)
            if not sol:
                continue
            sol_expr = sol[0][u]
            try:
                val = float(sol_expr.subs(subs).evalf())
                results[str(u)] = val
            except Exception:
                pass
        return results, metadata

# -----------------------
# Explanation builder
# -----------------------
def build_explanation(formula_text: str, expr: sp.Expr, given_q: Dict[str, Q_], result_vals: Dict[str, float]):
    lines = []
    lines.append("Formula:")
    lines.append(f"  {formula_text}")
    lines.append("\nVariables and given values (converted to SI base units where applicable):")
    for name, q in given_q.items():
        try:
            q_si = q.to_base_units()
            lines.append(f"  {name} = {q_si.magnitude} {q_si.units}")
        except Exception:
            lines.append(f"  {name} = {q}")
    lines.append("\nComputation:")
    for var, val in result_vals.items():
        lines.append(f"  {var} = {val} (SI units where applicable)")
    return "\n".join(lines)

# -----------------------
# Top-level function used by the UI
# -----------------------
def solve_from_formula_line(formula_line: str, situation: str, target: Optional[str] = None):
    if '=' not in formula_line:
        raise ValueError("Formula invalid (no '=')")
    lhs, rhs = [p.strip() for p in formula_line.split('=', 1)]
    expr = sp.sympify(lhs, evaluate=False) - sp.sympify(rhs, evaluate=False)
    variables = sorted(list(expr.free_symbols), key=lambda x: str(x))
    found = extract_values_from_text(situation, variables)
    subs_vals = prepare_subs(expr, found)
    if target is None:
        unknowns = [str(s) for s in variables if s not in subs_vals]
        if len(unknowns) == 1:
            target = unknowns[0]
        else:
            target = None
    results, metadata = solve_formula(expr, target, subs_vals)
    explanation = build_explanation(formula_line, expr, found, results)
    return results, metadata, explanation
