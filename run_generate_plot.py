import asyncio
import traceback

def run():
    try:
        import app_fixed
    except Exception as e:
        print('Failed to import app_fixed:', e)
        traceback.print_exc()
        return 1

    async def main():
        try:
            res = await app_fixed.generate_plot()
            print('generate_plot completed:', res)
        except Exception:
            traceback.print_exc()

    asyncio.run(main())
    return 0

if __name__ == '__main__':
    raise SystemExit(run())
import asyncio
import app_fixed

async def main():
    try:
        res = await app_fixed.generate_plot()
        print('generate_plot completed:', res)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
